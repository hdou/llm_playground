import re
from typing import List

class AgendaFragment:
    def __init__(self, time: str, location: str, content: str):
        self.time = time
        self.location = location
        self.content = content

    def __repr__(self):
        return (f"AgendaFragment(time={self.time!r}, location={self.location!r}, "
                f"content={self.content[:30]!r}...)")

class AgendaParser:
    TIME_LOC_PATTERN = re.compile(r"(\d{1,2}:\d{2} (?:AM|PM) - \d{1,2}:\d{2} (?:AM|PM) \(EDT\)) ([^\n]+)")

    def __init__(self, text: str):
        self.text = text
        self.fragments: List[AgendaFragment] = []

    def parse(self):
        matches = list(self.TIME_LOC_PATTERN.finditer(self.text))
        for i, match in enumerate(matches):
            start = match.end()
            end = matches[i+1].start() if i+1 < len(matches) else len(self.text)
            segment = self.text[start:end].strip()
            time = match.group(1)
            location = match.group(2)
            content = self._extract_content(segment)
            self.fragments.append(AgendaFragment(time, location, content))
        return self.fragments

    def _extract_content(self, segment: str):
        # Just return the remaining text as content
        return segment.strip()

if __name__ == "__main__":
    with open("agenda-pdf.txt", "r", encoding="utf-8") as f:
        text = f.read()
    parser = AgendaParser(text)
    fragments = parser.parse()
    for frag in fragments[:5]:  # Print first 5 for demo
        print(f"Time: {frag.time}")
        print(f"Location: {frag.location}")
        print("Content:")
        print(frag.content)
        print("\n" + "="*60 + "\n")

    for frag in fragments:
        print(f"length: {len(frag.content)}")
