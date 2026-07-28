import { AlbumSummary } from "../api/types";
import ComboBox from "./ComboBox";

/**
 * The picked set, and the small number of things you can do with it.
 *
 * The tray exists only while something is picked — a tray of zero would be
 * furniture — so the emptiness check lives here rather than at every call
 * site. Fixed to the viewport bottom so the running count and the album action
 * stay in reach however deep the scroll, which is exactly when hand-picking
 * happens.
 *
 * It owns no state: the picked set, the album name and the busy flag all
 * belong to whoever is doing the picking. This is a piece of UI over that
 * data, not a second place where a selection could live.
 */
export default function SelectionTray({
  picked, albums, albumName, onAlbumName, busy, onSave, onCompare, onClear,
}: {
  picked: Set<number>;
  /** Existing albums, so "add to an existing album" is one keystroke. */
  albums: AlbumSummary[];
  albumName: string;
  onAlbumName: (v: string) => void;
  /** A save is in flight: the primary action names it and refuses a second. */
  busy: boolean;
  onSave: () => void;
  /** Called with the two picked ids, in picked order. */
  onCompare: (a: number, b: number) => void;
  onClear: () => void;
}) {
  if (picked.size === 0) return null;

  return (
    <div className="selection-tray" role="toolbar" aria-label="Selected images">
      {/* The running count is the tray's subject, so the figure is set in
          the mono face with tabular figures: 2, 30 and 200 have to be
          readable at a glance and must not shift the controls beside them. */}
      <span className="select-count" aria-live="polite">
        <b className="select-n">{picked.size}</b> picked
      </span>
      {/* The album field is the product's own control, not the browser's:
          on the tray's dark ink a default input reads as a hole. The label
          rides inside the field so the control names itself without
          spending a row. */}
      <label className="tray-field">
        <span className="tray-field-label">Album</span>
        <ComboBox value={albumName} onChange={onAlbumName}
                  options={albums.map((a) => a.name)}
                  placeholder="new or existing…"
                  label="Album name"
                  onSubmit={() => { if (albumName.trim() && !busy) onSave(); }} />
      </label>
      <button className="primary" disabled={!picked.size || !albumName.trim() || busy}
              onClick={onSave}>
        {busy ? "Saving…" : `Add ${picked.size} to album`}
      </button>
      {/* The two exits travel together, so a tray narrow enough to wrap
          breaks between "file this set" and "leave it" instead of
          stranding Clear on a row of its own. */}
      <span className="tray-acts">
        {/* Compare is defined on a pair — one loupe, two images, one
            shared transform. It used to disappear at three picked, which
            reads as a bug rather than as a rule; it now stays put and says
            what it wants instead. */}
        <span className="tray-compare">
          <button className="ghost" disabled={picked.size !== 2}
                  title={picked.size === 2
                    ? "Open these two side by side with synchronized zoom"
                    : "Compare puts exactly two images under one loupe — "
                      + `${picked.size} picked`}
                  onClick={() => {
                    const [a, b] = [...picked];
                    onCompare(a, b);
                  }}>
            Compare
          </button>
          {picked.size !== 2 && (
            <span className="tray-rule">needs exactly 2</span>
          )}
        </span>
        {/* Clearing the set is also how the tray leaves: with no mode to
            exit, an emptied selection has nothing left to dismiss. */}
        <button className="ghost tray-clear" onClick={onClear}>
          Clear
        </button>
      </span>
    </div>
  );
}
