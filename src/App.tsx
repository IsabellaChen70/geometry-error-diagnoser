import { brainlift } from "./data/brainlift"
import { useScrollSpy } from "./lib/useScrollSpy"
import Sidebar from "./components/Sidebar"
import Masthead from "./components/Masthead"
import Scope from "./components/Scope"
import Spov from "./components/Spov"
import Insights from "./components/Insights"
import KnowledgeTree from "./components/KnowledgeTree"
import Experts from "./components/Experts"
import SiteFooter from "./components/SiteFooter"

const SECTION_IDS = ["purpose", "scope", "spov", "insights", "q1", "q2", "q3", "experts"]

export default function App() {
  const active = useScrollSpy(SECTION_IDS)
  const { meta, purpose, scope, spov, insights, knowledgeTree, collapseChart, experts, links } = brainlift

  return (
    <div className="layout">
      <Sidebar meta={meta} questions={knowledgeTree.questions} active={active} />
      <main className="main">
        <div className="main-inner">
          <Masthead meta={meta} purpose={purpose} />
          <Scope scope={scope} />
          <Spov spov={spov} />
          <Insights insights={insights} />
          <KnowledgeTree tree={knowledgeTree} chart={collapseChart} />
          <Experts experts={experts} />
          <SiteFooter meta={meta} links={links} />
        </div>
      </main>
    </div>
  )
}
