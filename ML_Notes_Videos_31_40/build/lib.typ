// Shared book template for the Video 31-40 notes.

#let primary = rgb("#1F3A5F")
#let secondary = rgb("#2E5E8C")
#let accent = rgb("#B03A2E")
#let ink = rgb("#1B1B1B")
#let muted = rgb("#5D6D7E")
#let hairline = rgb("#D5DBDB")
#let panel = rgb("#F4F6F8")
#let panel-blue = rgb("#EAF2F8")
#let panel-green = rgb("#EAF7EE")
#let panel-amber = rgb("#FDF6E3")
#let panel-red = rgb("#FBEDEC")
#let green = rgb("#1E7B45")
#let amber = rgb("#9A6700")

#let chapter-state = state("chapter-state", (num: 0, title: ""))

#let set-chapter(num, title) = chapter-state.update((num: num, title: title))

#let callout(title, body, fill: panel-blue, edge: primary, title-fill: primary) = block(
  width: 100%,
  breakable: true,
  inset: (x: 12pt, y: 9pt),
  radius: 2pt,
  fill: fill,
  stroke: (left: 3pt + edge),
  [
    #text(weight: "bold", fill: title-fill, size: 10pt)[#title]
    #v(3pt)
    #body
  ],
)

#let keybox(title: "Key idea", body) = callout(title, body, fill: panel-green, edge: green, title-fill: green)

#let warnbox(title: "Watch out", body) = callout(title, body, fill: panel-red, edge: accent, title-fill: accent)

#let notebox(title: "Note", body) = callout(title, body, fill: panel-amber, edge: amber, title-fill: amber)

#let examplebox(title: "Worked example", body) = callout(title, body, fill: panel-blue, edge: secondary, title-fill: secondary)

#let definition(term, body) = block(
  width: 100%,
  breakable: true,
  inset: (x: 12pt, y: 9pt),
  fill: panel,
  stroke: (left: 2pt + muted),
  [
    #text(weight: "bold", fill: primary)[#term]
    #v(2pt)
    #body
  ],
)

#let lead(body) = block(above: 0.4em, below: 0.9em, text(size: 11.5pt, fill: rgb("#34495E"), body))

#let chapter-divider(num, title, subtitle, summary) = {
  set-chapter(num, title)
  pagebreak()
  block(above: 0.4cm, below: 0.1cm)[
    #text(size: 10pt, weight: "bold", fill: accent, tracking: 2.5pt)[CHAPTER #num]
  ]
  heading(level: 1, outlined: true, bookmarked: true, title)
  block(above: 0.25cm, below: 0.55cm)[
    #text(size: 14pt, fill: secondary, weight: "bold")[#subtitle]
  ]
  block(
    width: 100%,
    inset: (x: 14pt, y: 11pt),
    fill: panel,
    stroke: (left: 3pt + primary),
  )[
    #text(size: 9.5pt, weight: "bold", fill: primary, tracking: 1pt)[WHAT THIS CHAPTER BUILDS]
    #v(4pt)
    #text(size: 10.5pt, fill: ink)[#summary]
  ]
  v(0.35cm)
}

#let book(
  title: "",
  subtitle: "",
  chapter: none,
  cover: true,
  toc: false,
  body,
) = {
  set document(title: title, author: "CampusX study notes")
  set page(
    paper: "a4",
    margin: (top: 2.35cm, bottom: 2.15cm, x: 2.25cm),
    header: context {
      let c = chapter-state.get()
      set text(size: 8pt, fill: muted)
      set par(justify: false)
      grid(
        columns: (1fr, auto),
        align: (left + horizon, right + horizon),
        if c.num == 0 { "" } else { [Chapter #c.num · #c.title] },
        [Machine Learning · Videos 31--40],
      )
      v(-2pt)
      line(length: 100%, stroke: 0.4pt + hairline)
    },
    footer: context {
      set text(size: 8.5pt, fill: muted)
      set align(center)
      counter(page).display()
    },
  )
  set text(font: ("Cambria", "Georgia"), size: 10.5pt, fill: ink, lang: "en")
  set par(justify: true, leading: 0.72em)
  set heading(numbering: none)

  show heading.where(level: 1): it => block(above: 1.1em, below: 0.35em)[
    #text(size: 27pt, weight: "bold", fill: primary, it.body)
  ]
  show heading.where(level: 2): it => block(above: 1.35em, below: 0.45em)[
    #text(size: 15pt, weight: "bold", fill: primary, it.body)
    #v(-1pt)
    #line(length: 100%, stroke: 0.7pt + accent)
  ]
  show heading.where(level: 3): it => block(above: 1.05em, below: 0.25em)[
    #text(size: 11.5pt, weight: "bold", fill: rgb("#2C3E50"), it.body)
  ]
  set list(indent: 1.15em, spacing: 0.55em)
  set enum(indent: 1.3em, spacing: 0.55em)

  show raw.where(block: false): it => box(
    fill: rgb("#F1F3F5"),
    inset: (x: 3pt, y: 1pt),
    radius: 1.5pt,
    text(font: "Consolas", size: 9pt, it),
  )
  show raw.where(block: true): it => block(
    width: 100%,
    breakable: true,
    fill: rgb("#F7F8FA"),
    stroke: 0.5pt + hairline,
    inset: (x: 11pt, y: 9pt),
    radius: 2pt,
    text(font: "Consolas", size: 8.6pt, it),
  )
  show figure.caption: it => block(above: 0.35em, below: 0.9em)[
    #set text(size: 9pt, fill: muted)
    #set par(justify: false)
    #it.body
  ]
  show table.cell.where(y: 0): set text(fill: white, weight: "bold", size: 9pt)
  show table: set text(size: 9.2pt)

  if chapter != none {
    set-chapter(chapter.num, chapter.title)
  }

  if cover {
    page(header: none, footer: none, margin: (x: 2.4cm, y: 2.4cm))[
      #set align(left)
      #v(2.5cm)
      #text(size: 9pt, weight: "bold", fill: accent, tracking: 3pt)[CAMPUSX · 100 DAYS OF MACHINE LEARNING]
      #v(0.45cm)
      #text(size: 33pt, weight: "bold", fill: primary)[#title]
      #v(0.15cm)
      #text(size: 18pt, fill: secondary)[#subtitle]
      #v(0.55cm)
      #line(length: 38%, stroke: 2.2pt + accent)
      #v(0.55cm)
      #text(size: 11pt, fill: ink)[Feature engineering and missing-data notes · Chapters 31--40]
      #v(0.18cm)
      #text(size: 10pt, fill: muted)[Rebuilt as a book: mechanism first, prose before lists, every example traced]
      #v(1fr)
      #text(size: 9pt, fill: muted)[Study notes from the CampusX 100 Days of Machine Learning playlist. Every claim is tied to the lecture, Scikit-learn behaviour, or a worked calculation.]
    ]
  }

  if toc {
    pagebreak()
    heading(level: 2, outlined: false, [Contents])
    outline(title: none, depth: 2, indent: auto)
  }

  body
}
