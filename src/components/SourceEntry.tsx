import type { Source } from "../types"
import { renderInline } from "../lib/inline"
import Chip from "./Chip"

interface Props {
  source: Source
  panelId: string
  open: boolean
  onToggle: () => void
}

export default function SourceEntry({ source, panelId, open, onToggle }: Props) {
  return (
    <article className="source">
      <div className="source-cite">
        <cite className="source-title">{renderInline(source.title)}</cite>
        <p className="source-meta">
          {source.authors ? <span className="cite-authors">{source.authors}</span> : null}
          {source.venue ? <span className="cite-venue mono">{source.venue}</span> : null}
          {source.institution ? <span className="cite-inst">{source.institution}</span> : null}
        </p>
        {source.links.length > 0 ? (
          <div className="chips">
            {source.links.map((link, i) => (
              <Chip key={i} link={link} />
            ))}
          </div>
        ) : null}
      </div>

      <div className="dok">
        <p className="dok-label mono">DOK 1 · Facts</p>
        <ul className="facts">
          {source.facts.map((fact, i) =>
            fact.kind === "quote" ? (
              <li key={i} className="fact fact-quote">
                <blockquote>{renderInline(fact.text)}</blockquote>
              </li>
            ) : (
              <li key={i} className="fact">
                {renderInline(fact.text)}
              </li>
            )
          )}
        </ul>
      </div>

      <div className="dok dok-analysis">
        <button
          type="button"
          className="dok-toggle"
          aria-expanded={open}
          aria-controls={panelId}
          onClick={onToggle}
        >
          <span className="dok-label mono">DOK 2 · Analysis</span>
          <span className="chev" data-open={open} aria-hidden="true" />
        </button>
        <div id={panelId} className={open ? "dok-panel open" : "dok-panel"} aria-hidden={!open}>
          <div className="dok-panel-inner">
            <p>{renderInline(source.analysis)}</p>
          </div>
        </div>
      </div>
    </article>
  )
}
