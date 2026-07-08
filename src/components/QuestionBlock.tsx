import { useState, type ReactNode } from "react"
import type { Flag, Question } from "../types"
import SourceEntry from "./SourceEntry"

function FlagTag({ flag }: { flag: Flag }) {
  const label = flag === "verified" ? "Verified finding" : "Ruled out"
  return <span className={`flag flag-${flag}`}>{label}</span>
}

export default function QuestionBlock({ question, topFigure }: { question: Question; topFigure?: ReactNode }) {
  const keys: string[] = []
  question.subnodes.forEach((sn, si) => sn.sources.forEach((_, ci) => keys.push(`${si}-${ci}`)))

  const [open, setOpen] = useState<Set<string>>(() => new Set())
  const allOpen = keys.length > 0 && open.size === keys.length

  const toggleOne = (key: string) =>
    setOpen((prev) => {
      const next = new Set(prev)
      if (next.has(key)) next.delete(key)
      else next.add(key)
      return next
    })

  const toggleAll = () => setOpen(allOpen ? new Set() : new Set(keys))

  return (
    <section id={question.id} className="question" aria-labelledby={`${question.id}-h`}>
      <div className="question-head">
        <p className="question-n">{question.n.padStart(2, "0")}</p>
        <h3 id={`${question.id}-h`} className="question-title">
          {question.title}
        </h3>
        <button type="button" className="expand-all" onClick={toggleAll}>
          {allOpen ? "Collapse all" : "Expand all"}
        </button>
      </div>

      {topFigure}

      {question.subnodes.map((sn, si) => (
        <div className="subnode" key={sn.n}>
          <div className="subnode-head">
            <span className="subnode-n mono">{sn.n}</span>
            <h4 className="subnode-title">{sn.title}</h4>
            {sn.flag ? <FlagTag flag={sn.flag} /> : null}
          </div>
          <div className="sources">
            {sn.sources.map((src, ci) => {
              const key = `${si}-${ci}`
              return (
                <SourceEntry
                  key={key}
                  source={src}
                  panelId={`${question.id}-${key}`}
                  open={open.has(key)}
                  onToggle={() => toggleOne(key)}
                />
              )
            })}
          </div>
        </div>
      ))}
    </section>
  )
}
