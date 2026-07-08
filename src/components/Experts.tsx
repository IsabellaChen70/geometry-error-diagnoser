import type { Expert } from "../types"
import { renderInline } from "../lib/inline"
import Chip from "./Chip"

export default function Experts({ experts }: { experts: Expert[] }) {
  return (
    <section id="experts" className="experts" aria-labelledby="experts-h">
      <div className="section-head">
        <h2 id="experts-h" className="section-title">Experts</h2>
      </div>
      <ul className="expert-list">
        {experts.map((expert, i) => (
          <li className="expert" key={i}>
            <div className="expert-id">
              <h3 className="expert-name">{expert.name}</h3>
              {expert.institution ? <p className="expert-inst mono">{expert.institution}</p> : null}
            </div>
            <div className="expert-body">
              <p className="expert-views">
                <span className="micro-label">Main views</span>
                {renderInline(expert.views)}
              </p>
              <p className="expert-why">
                <span className="micro-label">Why follow</span>
                {renderInline(expert.whyFollow)}
              </p>
              {expert.links.length > 0 ? (
                <div className="chips">
                  {expert.links.map((link, j) => (
                    <Chip key={j} link={link} />
                  ))}
                </div>
              ) : null}
            </div>
          </li>
        ))}
      </ul>
    </section>
  )
}
