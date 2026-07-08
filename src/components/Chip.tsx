import type { Link } from "../types"

export default function Chip({ link }: { link: Link }) {
  if (link.url) {
    return (
      <a className="chip" href={link.url} target="_blank" rel="noreferrer">
        <span className="chip-label">{link.label}</span>
        <span className="chip-ext" aria-hidden="true">↗</span>
      </a>
    )
  }
  return <span className="chip chip-static">{link.label}</span>
}
