import { useState } from "react"
import { demoExamples, type DemoExample, type NetMap } from "../data/demo"

const LINEAR_DESCRIPTIONS: Record<string, string> = {
  identity: "identity",
  rot_ccw_90: "90\u00b0 rotation (ccw)",
  rot_180: "180\u00b0 rotation",
  rot_ccw_270: "270\u00b0 rotation (ccw)",
  reflect_x_axis: "reflect across the x-axis",
  reflect_y_axis: "reflect across the y-axis",
  reflect_y_eq_x: "reflect across y = x",
  reflect_y_eq_neg_x: "reflect across y = -x",
}

function labelText(label: string): string {
  const words = label.replace(/_/g, " ")
  return words.charAt(0).toUpperCase() + words.slice(1)
}

function describeNet(net: NetMap): string {
  const linear = LINEAR_DESCRIPTIONS[net.linear] ?? net.linear
  const translation =
    net.tx === 0 && net.ty === 0 ? "no translation" : `translate (${net.tx}, ${net.ty})`
  return `${linear}, then ${translation}`
}

function netJson(net: NetMap): string {
  return `{ "linear": "${net.linear}", "tx": ${net.tx}, "ty": ${net.ty} }`
}

function MapRow({ tag, flow, net }: { tag: string; flow: string; net: NetMap }) {
  return (
    <div className="demo-map">
      <p className="micro-label">
        {tag} <span className="demo-flow">{flow}</span>
      </p>
      <p className="demo-map-desc">{describeNet(net)}</p>
      <p className="demo-net mono">{netJson(net)}</p>
    </div>
  )
}

export default function Demo() {
  const [index, setIndex] = useState(0)
  const example: DemoExample = demoExamples[index]
  const isCorrect = example.label === "correct"

  return (
    <section id="demo" className="demo" aria-labelledby="demo-h">
      <div className="section-head">
        <h2 id="demo-h" className="section-title">
          See the model work
        </h2>
      </div>

      <p className="demo-intro">
        Pick a diagram. Each shows a RED original polygon, the GREEN correct image,
        and the BLUE student image, alongside the diagnosis the model is trained to
        return: the two net affine maps, the error label those maps imply, and a
        hint. Examples are drawn from the v6 sample set; on held-out test the tuned
        image+coordinates model recovers both maps on 98.6% of cases.
      </p>

      <div className="demo-controls">
        <label htmlFor="demo-select" className="micro-label">
          Example
        </label>
        <select
          id="demo-select"
          className="demo-select mono"
          value={index}
          onChange={(event) => setIndex(Number(event.target.value))}
        >
          {demoExamples.map((demo, i) => (
            <option key={demo.id} value={i}>
              {labelText(demo.label)}
            </option>
          ))}
        </select>
      </div>

      <div className="demo-grid">
        <figure className="demo-figure">
          <img
            className="demo-img"
            src={example.imageUrl}
            width={656}
            height={640}
            alt={`Coordinate-grid diagram: a RED original polygon with ${example.numVertices} numbered vertices, the GREEN correct image, and the BLUE student image.`}
          />
        </figure>

        <div className="demo-diagnosis">
          <div className="demo-diagnosis-head">
            <span className="micro-label">Diagnosis</span>
            <span className={isCorrect ? "demo-label-badge is-correct" : "demo-label-badge"}>
              {labelText(example.label)}
            </span>
          </div>

          <div className="demo-maps">
            <MapRow tag="Correct map" flow={"RED \u2192 GREEN"} net={example.correctNet} />
            <MapRow tag="Student map" flow={"RED \u2192 BLUE"} net={example.studentNet} />
          </div>

          <div className="demo-hint">
            <p className="micro-label">Hint</p>
            <p className="demo-hint-text">{example.hint}</p>
          </div>
        </div>
      </div>

      <p className="demo-note">
        These are the canonical target diagnoses for sample diagrams, scored by the
        deterministic geometry oracle rather than produced by a live model here. The
        hints are the training targets and can state the answer; see the hint-safety
        note in the README.
      </p>
    </section>
  )
}
