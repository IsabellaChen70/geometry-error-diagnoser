import type { Insight } from "../types"
import { renderInline } from "../lib/inline"

export default function Insights({ insights }: { insights: Insight[] }) {
  return (
    <section id="insights" className="insights" aria-labelledby="insights-h">
      <div className="section-head">
        <h2 id="insights-h" className="section-title">Insights</h2>
        <span className="section-label mono">DOK 3</span>
      </div>
      <ol className="insight-list">
        {insights.map((insight) => (
          <li className="insight" key={insight.n}>
            <div className="insight-n mono">{insight.n}</div>
            <div className="insight-body">
              <h3 className="insight-title">{renderInline(insight.title)}</h3>
              <p className="insight-text">{renderInline(insight.body)}</p>
              <p className="insight-draws">
                <span className="draws-label mono">Draws on</span>
                <span className="draws-list">{renderInline(insight.drawsOn)}</span>
              </p>
            </div>
          </li>
        ))}
      </ol>
    </section>
  )
}
