import type { Brainlift } from "../types"
import { renderInline } from "../lib/inline"

export default function Scope({ scope }: { scope: Brainlift["scope"] }) {
  return (
    <section id="scope" className="scope" aria-labelledby="scope-h">
      <div className="section-head">
        <h2 id="scope-h" className="section-title">Scope</h2>
      </div>
      <div className="scope-grid">
        <div className="scope-block">
          <h3 className="scope-sub">In scope</h3>
          <ul className="scope-list">
            {scope.inScope.map((item, i) => (
              <li key={i}>{renderInline(item)}</li>
            ))}
          </ul>
        </div>
        <div className="scope-block">
          <h3 className="scope-sub">Out of scope</h3>
          <p className="scope-tbd">To be defined.</p>
        </div>
      </div>
    </section>
  )
}
