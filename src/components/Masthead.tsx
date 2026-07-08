import type { Meta } from "../types"
import CompositionFigure from "./CompositionFigure"

export default function Masthead({ meta, purpose }: { meta: Meta; purpose: string }) {
  return (
    <header id="purpose" className="masthead">
      <p className="eyebrow">{meta.kind}</p>
      <h1 className="masthead-title">{meta.title}</h1>
      <p className="lead">{purpose}</p>

      <dl className="meta-row">
        <div className="meta-item">
          <dt>Owner</dt>
          <dd>{meta.owner}</dd>
        </div>
        <div className="meta-item">
          <dt>Standard</dt>
          <dd className="mono">{meta.standard}</dd>
        </div>
        <div className="meta-item">
          <dt>Sources</dt>
          <dd className="mono">{meta.sourceCount}</dd>
        </div>
        <div className="meta-item">
          <dt>Compiled</dt>
          <dd className="mono">{meta.compiled}</dd>
        </div>
      </dl>

      <CompositionFigure />
    </header>
  )
}
