import type { Brainlift } from "../types"
import QuestionBlock from "./QuestionBlock"
import CollapseChart from "./CollapseChart"

interface Props {
  tree: Brainlift["knowledgeTree"]
  chart: Brainlift["collapseChart"]
}

export default function KnowledgeTree({ tree, chart }: Props) {
  return (
    <section className="tree" aria-labelledby="tree-h">
      <div className="section-head">
        <h2 id="tree-h" className="section-title">Knowledge tree</h2>
        <span className="section-label mono">{tree.label}</span>
      </div>
      <div className="questions">
        {tree.questions.map((q) => (
          <QuestionBlock
            key={q.id}
            question={q}
            topFigure={q.id === "q1" ? <CollapseChart rows={chart} /> : null}
          />
        ))}
      </div>
    </section>
  )
}
