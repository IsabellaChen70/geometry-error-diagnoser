import { useEffect, useState } from "react"

// Highlights the table-of-contents entry for the section currently in view.
// Uses IntersectionObserver instead of a scroll listener, so there is no
// per-scroll layout thrash from reading getBoundingClientRect on every section.
// A negative rootMargin turns the top of the viewport into a thin activation
// band: a section counts as "in view" once its top scrolls under the band, and
// the active entry is the first such section in document order.
export function useScrollSpy(ids: string[], offset = 96): string {
  const [active, setActive] = useState(ids[0] ?? "")

  useEffect(() => {
    const elements = ids
      .map((id) => document.getElementById(id))
      .filter((el): el is HTMLElement => el !== null)
    if (elements.length === 0) return

    const visible = new Set<string>()
    const observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (entry.isIntersecting) visible.add(entry.target.id)
          else visible.delete(entry.target.id)
        }
        const current = ids.find((id) => visible.has(id))
        if (current) setActive(current)
      },
      { rootMargin: `-${offset}px 0px -55% 0px`, threshold: 0 }
    )

    elements.forEach((el) => observer.observe(el))
    return () => observer.disconnect()
  }, [ids, offset])

  return active
}
