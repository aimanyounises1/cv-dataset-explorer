import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import App from "./App";
// panels.css before index.css, because index.css must win ties. Several panels
// restyle a shared primitive at the same specificity — `.cmdk-loading` against
// `.loading`, `.cmdk-group-count` against `.pill`, `.saved-views-error` against
// `.error` — so whichever file comes last decides those, and last is where the
// primitives have always been. The panels used to import their own stylesheets,
// which put all of them ahead of index.css by accident of the module graph;
// naming the order here keeps the rendering identical and stops it depending on
// which component happens to be imported first.
import "./styles/panels.css";
import "./index.css";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <BrowserRouter>
      <App />
    </BrowserRouter>
  </React.StrictMode>,
);
