# CV Dataset Explorer UI: `npm run build` output served by nginx, with the
# same two prefixes the Vite dev server proxies pointed at the API container.
#
# The frontend reads no build-time configuration — every request it makes is a
# relative /api or /media path — so one image is correct for any host the
# stack runs on, and there is no VITE_* variable to bake in or get wrong.
#
# Build context is the repository root, because the image needs both
# frontend/ and docker/nginx.conf.

FROM node:22-slim AS build

WORKDIR /src

# `npm ci` installs from the lockfile exactly, and lands in its own layer so
# it is only redone when the manifests change rather than on every source edit.
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend/ ./
# `npm run build` is `tsc && vite build`: a type error fails the image build,
# which is the same bar the host workflow applies.
RUN npm run build


FROM nginx:alpine

COPY --from=build /src/dist /usr/share/nginx/html
COPY docker/nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

# wget is in nginx:alpine's busybox already, so this adds no layer.
#
# 127.0.0.1 and not localhost, which is not cosmetic: nginx listens on
# 0.0.0.0:80 (IPv4 only), busybox wget resolves localhost to ::1 first and does
# not fall back, so the `localhost` form fails with "connection refused" against
# a container that is serving perfectly. curl's happy-eyeballs fallback hides
# the same problem, which is why this is worth a comment rather than a fix in
# passing.
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD wget -q --spider http://127.0.0.1/ || exit 1
