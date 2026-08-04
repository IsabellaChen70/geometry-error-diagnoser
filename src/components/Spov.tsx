import type { Spov as SpovData } from "../types"
import { renderInline } from "../lib/inline"

export default function Spov({ spov }: { spov: SpovData }) {
  return (
    <section id="spov" className="spov" aria-labelledby="spov-h">
      <div className="section-head">
        <h2 id="spov-h" className="section-title">Key claim</h2>
      </div>

      <div className="spov-claim">
        <p className="micro-label">The claim</p>
        <p className="spov-claim-text">{renderInline(spov.claim)}</p>
      </div>

      <div className="spov-parts">
        <div className="spov-part">
          <p className="micro-label">Why it's non-obvious</p>
          <p className="spov-text">{renderInline(spov.whySpiky)}</p>
        </div>

        <div className="spov-part">
          <p className="micro-label">The argument</p>
          <p className="spov-text">{renderInline(spov.argument)}</p>
        </div>

        <div className="spov-part spov-bet">
          <p className="micro-label">
            The bet<span className="spov-anno">falsifiable prediction</span>
          </p>
          <p className="spov-text">{renderInline(spov.bet)}</p>
        </div>

        <div className="spov-part spov-caveat">
          <p className="micro-label">What would change our mind</p>
          <p className="spov-text">{renderInline(spov.changeMind)}</p>
        </div>
      </div>
    </section>
  )
}
