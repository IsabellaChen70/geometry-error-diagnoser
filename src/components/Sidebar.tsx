import { useState } from "react"
import type { Meta, Question } from "../types"

interface Props {
  meta: Meta
  questions: Question[]
  active: string
}

export default function Sidebar({ meta, questions, active }: Props) {
  const [open, setOpen] = useState(false)
  const close = () => setOpen(false)

  const items = [
    { id: "purpose", n: "", label: "Purpose" },
    { id: "scope", n: "", label: "Scope" },
    { id: "spov", n: "", label: "Spiky POV" },
    { id: "insights", n: "", label: "Insights" },
    ...questions.map((q) => ({ id: q.id, n: q.n.padStart(2, "0"), label: q.title })),
    { id: "experts", n: "", label: "Experts" },
  ]

  return (
    <aside className="sidebar">
      <div className="sidebar-top">
        <a className="brand" href="#purpose" onClick={close}>
          <span className="eyebrow">{meta.kind}</span>
          <span className="brand-title">{meta.title}</span>
        </a>
        <button
          type="button"
          className="contents-toggle"
          aria-expanded={open}
          aria-controls="toc"
          onClick={() => setOpen((v) => !v)}
        >
          Contents
        </button>
      </div>

      <dl className="sidebar-meta">
        <div>
          <dt>Owner</dt>
          <dd>{meta.owner}</dd>
        </div>
        <div>
          <dt>Standard</dt>
          <dd className="mono">{meta.standard}</dd>
        </div>
      </dl>

      <nav id="toc" className={open ? "toc open" : "toc"} aria-label="Sections">
        <ol>
          {items.map((item) => (
            <li key={item.id}>
              <a
                href={`#${item.id}`}
                className={active === item.id ? "toc-link active" : "toc-link"}
                aria-current={active === item.id ? "true" : undefined}
                onClick={close}
              >
                {item.n ? <span className="toc-n mono">{item.n}</span> : null}
                <span className="toc-label">{item.label}</span>
              </a>
            </li>
          ))}
        </ol>
      </nav>
    </aside>
  )
}
