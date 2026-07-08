import { Fragment, type ReactNode } from "react"

const TOKEN = /\*\*([^*]+)\*\*|\*([^*]+)\*/g

// Renders the small amount of inline emphasis carried in the research text
// (**strong** and *emphasis*) while passing every other character through
// untouched, so curly quotes, arrows, and em dashes stay verbatim.
export const renderInline = (text: string): ReactNode => {
  const nodes: ReactNode[] = []
  let last = 0
  let key = 0
  let match: RegExpExecArray | null
  TOKEN.lastIndex = 0
  while ((match = TOKEN.exec(text)) !== null) {
    if (match.index > last) {
      nodes.push(<Fragment key={key++}>{text.slice(last, match.index)}</Fragment>)
    }
    if (match[1] !== undefined) {
      nodes.push(<strong key={key++}>{match[1]}</strong>)
    } else {
      nodes.push(<em key={key++}>{match[2]}</em>)
    }
    last = TOKEN.lastIndex
  }
  if (last < text.length) {
    nodes.push(<Fragment key={key++}>{text.slice(last)}</Fragment>)
  }
  return nodes
}
