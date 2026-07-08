import type { ChartRow } from "../types"

export default function CollapseChart({ rows }: { rows: ChartRow[] }) {
  return (
    <figure className="chart">
      <div className="chart-plot">
        {rows.map((row) => (
          <div className="chart-row" key={row.benchmark}>
            <div className="chart-row-head">
              <span className="chart-benchmark">{row.benchmark}</span>
              <span className="chart-condition mono">{row.condition}</span>
            </div>
            <div className="chart-bars">
              <div className="chart-line">
                <span className="chart-tag">human</span>
                <div className="chart-track">
                  <div className="bar bar-human" style={{ width: `${row.human}%` }} aria-hidden="true" />
                </div>
                <span className="chart-value mono">{row.humanText}</span>
              </div>
              <div className="chart-line">
                <span className="chart-tag">models</span>
                <div className="chart-track">
                  <div
                    className="bar bar-model"
                    style={{ left: `${row.modelLow}%`, width: `${Math.max(row.modelHigh - row.modelLow, 0.8)}%` }}
                    aria-hidden="true"
                  />
                </div>
                <span className="chart-value chart-value-accent mono">{row.modelText}</span>
              </div>
            </div>
          </div>
        ))}
      </div>
      <figcaption className="chart-caption">
        Human baseline against model range on the composed condition, drawn from four benchmarks: Mind's Eye,
        TangramPuzzle, DORI, and LEGO-Puzzles. Values are accuracy from 0 to 100 percent.
      </figcaption>
    </figure>
  )
}
