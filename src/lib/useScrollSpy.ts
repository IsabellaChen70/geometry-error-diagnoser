import { useEffect, useState } from "react"

export function useScrollSpy(ids: string[], offset = 96): string {
  const [active, setActive] = useState(ids[0] ?? "")

  useEffect(() => {
    const update = () => {
      let current = ids[0] ?? ""
      for (const id of ids) {
        const el = document.getElementById(id)
        if (!el) continue
        if (el.getBoundingClientRect().top - offset <= 1) current = id
      }
      setActive(current)
    }
    update()
    window.addEventListener("scroll", update, { passive: true })
    window.addEventListener("resize", update)
    return () => {
      window.removeEventListener("scroll", update)
      window.removeEventListener("resize", update)
    }
  }, [ids, offset])

  return active
}
