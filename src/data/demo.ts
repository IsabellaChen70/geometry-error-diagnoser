import raw from "./demo.json"
import img000003 from "../../dataset_sample_v6/images/000003.png"
import img000000 from "../../dataset_sample_v6/images/000000.png"
import img000002 from "../../dataset_sample_v6/images/000002.png"
import img000005 from "../../dataset_sample_v6/images/000005.png"
import img000012 from "../../dataset_sample_v6/images/000012.png"

export interface NetMap {
  linear: string
  tx: number
  ty: number
}

export interface DemoRecord {
  id: number
  label: string
  image: string
  numVertices: number
  original: number[][]
  correctTransform: string[]
  studentTransform: string[]
  correctNet: NetMap
  studentNet: NetMap
  hint: string
}

export interface DemoExample extends DemoRecord {
  imageUrl: string
}

// Records are baked from dataset_sample_v6/train_v6.jsonl; the images are the
// matching rendered diagrams, imported so Vite fingerprints them and applies the
// site base path. Keep this map in sync with the "image" fields in demo.json.
const IMAGES: Record<string, string> = {
  "000003.png": img000003,
  "000000.png": img000000,
  "000002.png": img000002,
  "000005.png": img000005,
  "000012.png": img000012,
}

export const demoExamples: DemoExample[] = (raw as unknown as DemoRecord[]).map(
  (record) => ({ ...record, imageUrl: IMAGES[record.image] })
)
