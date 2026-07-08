export interface Link {
  label: string
  url?: string
}

export type FactKind = "fact" | "quote"

export interface Fact {
  kind: FactKind
  text: string
}

export interface Source {
  authors: string
  title: string
  venue: string
  institution: string
  facts: Fact[]
  analysis: string
  links: Link[]
}

export type Flag = "verified" | "negative"

export interface Subnode {
  n: string
  title: string
  flag?: Flag
  sources: Source[]
}

export interface Question {
  n: string
  id: string
  title: string
  subnodes: Subnode[]
}

export interface Expert {
  name: string
  institution: string
  views: string
  whyFollow: string
  links: Link[]
}

export interface ChartRow {
  benchmark: string
  condition: string
  human: number
  humanText: string
  modelLow: number
  modelHigh: number
  modelText: string
}

export interface Spov {
  claim: string
  whySpiky: string
  argument: string
  bet: string
  changeMind: string
}

export interface Insight {
  n: number
  title: string
  body: string
  drawsOn: string
}

export interface Meta {
  kind: string
  title: string
  owner: string
  ownerRole: string
  standard: string
  compiled: string
  sourceCount: number
  expertCount: number
}

export interface Brainlift {
  meta: Meta
  purpose: string
  scope: { inScope: string[]; outOfScope: string[] }
  spov: Spov
  insights: Insight[]
  knowledgeTree: { label: string; questions: Question[] }
  collapseChart: ChartRow[]
  experts: Expert[]
  links: Link[]
}
