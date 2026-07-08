const SHAPE = "0,0 96,0 96,32 32,32 32,96 0,96"

export default function CompositionFigure() {
  return (
    <figure className="composition">
      <svg
        className="composition-svg"
        viewBox="0 0 640 260"
        role="img"
        aria-label="A single polygon on a coordinate grid shown in three states: the original, then translated, then rotated."
      >
        <defs>
          <pattern id="grid" width="32" height="32" patternUnits="userSpaceOnUse">
            <path className="fig-grid" d="M 32 0 L 0 0 0 32" fill="none" />
          </pattern>
          <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
            <path className="fig-arrowhead" d="M 0 1 L 9 5 L 0 9" fill="none" />
          </marker>
        </defs>

        <rect className="fig-plot" x="16" y="20" width="608" height="216" fill="url(#grid)" />

        <polygon className="fig-orig" points={SHAPE} transform="translate(56,84)" />
        <circle className="fig-dot fig-dot-ink" cx="88" cy="116" r="3.2" />

        <polygon className="fig-ghost" points={SHAPE} transform="translate(232,60)" />
        <circle className="fig-dot fig-dot-muted" cx="264" cy="92" r="3.2" />

        <polygon className="fig-rot figure-draw" points={SHAPE} transform="translate(432,172) rotate(-90)" />
        <circle className="fig-dot fig-dot-accent" cx="464" cy="140" r="3.2" />

        <line className="fig-arrow" x1="160" y1="118" x2="222" y2="104" markerEnd="url(#arrow)" />
        <text className="fig-label" x="191" y="150" textAnchor="middle">translate</text>

        <path className="fig-arrow" d="M 336 112 Q 382 84 424 112" fill="none" markerEnd="url(#arrow)" />
        <text className="fig-label" x="380" y="150" textAnchor="middle">rotate</text>
      </svg>
    </figure>
  )
}
