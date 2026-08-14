from __future__ import annotations

import re


DEFAULT_WORDS_PER_PAGE = 300
DEFAULT_PAGES_PER_SEGMENT = 3


def _split_long_sentence(sentence: str, words_per_page: int) -> list[str]:
    words = sentence.split()
    if len(words) <= words_per_page:
        return [sentence]
    return [
        " ".join(words[index:index + words_per_page])
        for index in range(0, len(words), words_per_page)
    ]


def split_into_reading_pages(text: str, words_per_page: int = DEFAULT_WORDS_PER_PAGE) -> list[str]:
    if words_per_page <= 0:
        raise ValueError("words_per_page must be positive")
    cleaned = re.sub(r"https?://\S+|www\.\S+", "", text or "")
    sentences = re.split(r"(?<=[.?!。！？])\s+", cleaned)
    pages: list[str] = []
    current_sentences: list[str] = []
    current_word_count = 0

    def flush_page() -> None:
        nonlocal current_sentences, current_word_count
        page = " ".join(current_sentences).strip()
        if page:
            pages.append(page)
        current_sentences = []
        current_word_count = 0

    for raw_sentence in sentences:
        sentence = raw_sentence.strip()
        if not sentence:
            continue
        for part in _split_long_sentence(sentence, words_per_page):
            word_count = len(part.split())
            if current_sentences and current_word_count + word_count > words_per_page:
                flush_page()
            current_sentences.append(part)
            current_word_count += word_count
            if current_word_count >= words_per_page:
                flush_page()

    flush_page()
    return pages or ["내용이 없습니다."]


def group_pages(pages: list[str], pages_per_segment: int = DEFAULT_PAGES_PER_SEGMENT) -> list[list[dict]]:
    if pages_per_segment <= 0:
        raise ValueError("pages_per_segment must be positive")
    grouped: list[list[dict]] = []
    for start in range(0, len(pages), pages_per_segment):
        group = []
        for page_index in range(start, min(start + pages_per_segment, len(pages))):
            group.append({
                "page_index": page_index,
                "text": pages[page_index],
                "is_new_segment": page_index == start,
            })
        grouped.append(group)
    return grouped
