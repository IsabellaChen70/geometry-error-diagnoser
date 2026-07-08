import type { Brainlift } from "../types"
import Chip from "./Chip"

interface Props {
  meta: Brainlift["meta"]
  links: Brainlift["links"]
}

export default function SiteFooter({ meta, links }: Props) {
  return (
    <footer className="site-footer">
      <p className="footer-line">Brainlift compiled by {meta.owner}.</p>
      <p className="footer-note">Numbers, quotes, and wording are preserved from the original research.</p>
      <nav className="link-index" aria-label="All papers">
        {links.map((link, i) => (
          <Chip key={i} link={link} />
        ))}
      </nav>
    </footer>
  )
}
