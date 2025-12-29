import fitz  # PyMuPDF
import re
import os
import json
import random
import sys

# [1] NLTK 라이브러리 설정 (자동 키워드 확장용)
import nltk
from nltk.corpus import wordnet

# NLTK 데이터 다운로드 (최초 1회 실행 시 자동 다운로드)
try:
    nltk.data.find('corpora/wordnet.zip')
except LookupError:
    print("[Analyzer] NLTK WordNet 데이터 다운로드 중... (최초 1회)")
    try:
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
    except Exception as e:
        print(f"[Analyzer] NLTK 다운로드 실패: {e}")

# [2] defaults 경로 설정 및 시스템 경로 추가
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULTS_DIR = os.path.join(BASE_DIR, 'defaults')
if DEFAULTS_DIR not in sys.path:
    sys.path.append(DEFAULTS_DIR)
INDEX_FILE = os.path.join(DEFAULTS_DIR, 'music_index.json')


# =========================================================
# 3. 키워드 맵 자동 생성 (NLTK + EMOTIONS_20)
# =========================================================
def build_vibe_keywords_automatically():
    """
    defaults/emotions_20.py를 읽고 NLTK로 동의어를 확장하여
    분석용 키워드 맵을 생성합니다.
    """
    keyword_map = {}

    try:
        # 감정 정의 파일 임포트
        import emotions_20  # 파일명이 emotions_20.py라고 가정
        EMOTIONS_20 = emotions_20.EMOTIONS_20

        for key in EMOTIONS_20.keys():
            synonyms = set([key])
            try:
                for syn in wordnet.synsets(key):
                    for lemma in syn.lemmas():
                        word = lemma.name().replace('_', ' ').lower()
                        synonyms.add(word)
            except:
                pass

            style_desc = EMOTIONS_20[key].get("style", "")
            if style_desc:
                desc_words = [w.strip().lower() for w in style_desc.split(',')]
                synonyms.update(desc_words)

            keyword_map[key] = list(synonyms)

    except ImportError:
        # 파일이 없으면 기본값 사용
        return {
            "joy": ["happy", "smile", "fun", "excited", "delight"],
            "sadness": ["sad", "cry", "tear", "grief", "depression"],
            "fear": ["scary", "dark", "ghost", "horror", "afraid"],
            "anger": ["angry", "mad", "rage", "furious"],
            "neutral": ["normal", "calm", "quiet", "silent"],
            "surprise": ["shock", "amazed", "wow", "sudden"]
        }
    except Exception as e:
        print(f"[Analyzer] ⚠️ 키워드 생성 중 오류: {e}")
        return {"neutral": ["neutral"]}

    return keyword_map


# 전역 변수로 키워드 맵 생성
TEXT_VIBE_KEYWORDS = build_vibe_keywords_automatically()


# =========================================================
# 4. 음악 매퍼 클래스
# =========================================================
class MusicMapper:
    def __init__(self):
        self.raw_library = {}
        self.genre_bucket = {}
        self._load_index()

    def _load_index(self):
        if os.path.exists(INDEX_FILE):
            try:
                with open(INDEX_FILE, 'r', encoding='utf-8') as f:
                    self.raw_library = json.load(f)
                self._build_genre_map()
            except Exception as e:
                print(f"[Analyzer] 인덱스 로드 실패: {e}")
        else:
            pass

    def _build_genre_map(self):
        self.genre_bucket = {}
        for file_id, info in self.raw_library.items():
            vibe_key = info.get('genre') or info.get('vibe') or 'unknown'
            vibe_key = vibe_key.lower()

            if vibe_key not in self.genre_bucket:
                self.genre_bucket[vibe_key] = []
            self.genre_bucket[vibe_key].append(info)

    def analyze_vibe(self, text):
        if not text: return "neutral"
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)
        scores = {vibe: 0 for vibe in TEXT_VIBE_KEYWORDS}

        for word in words:
            for vibe, keywords in TEXT_VIBE_KEYWORDS.items():
                if word in keywords:
                    scores[vibe] += 1

        if not any(scores.values()):
            return "neutral"
        return max(scores, key=scores.get)

    def get_music(self, vibe):
        vibe = vibe.lower()

        # 1) Exact match
        if vibe in self.genre_bucket and self.genre_bucket[vibe]:
            return random.choice(self.genre_bucket[vibe])

        # 2) Emotion -> genre category mapping
        emotion_to_categories = {
            "anger": ["industrial", "noise", "battle", "rock", "trap", "hiphop", "techno", "metal"],
            "disgust": ["industrial", "noise", "horror", "drone"],
            "fear": ["horror", "noir", "drone", "ambient", "suspense"],
            "joy": ["pop", "edm", "house", "funk", "samba", "reggae", "trance", "groove", "disco"],
            "neutral": ["minimal", "ambient", "lofi", "jazz", "acoustic", "drone", "calm"],
            "sadness": ["ambient", "lofi", "noir", "acoustic", "romantic", "impressionist", "blues"],
            "surprise": ["experimental", "vaporwave", "noise", "industrial", "fusion", "synthwave"],
        }

        if vibe in emotion_to_categories:
            for cat in emotion_to_categories[vibe]:
                if cat in self.genre_bucket and self.genre_bucket[cat]:
                    return random.choice(self.genre_bucket[cat])

        # 3) Alias mapping
        alias = {
            "peace": "neutral", "love": "romantic", "hope": "ambient",
            "curiosity": "experimental", "confusion": "experimental",
            "courage": "battle", "pride": "cinematic", "excitement": "joy"
        }
        if vibe in alias:
            alt = alias[vibe]
            if alt in self.genre_bucket and self.genre_bucket[alt]:
                return random.choice(self.genre_bucket[alt])
            if alt in emotion_to_categories:
                for cat in emotion_to_categories[alt]:
                    if cat in self.genre_bucket and self.genre_bucket[cat]:
                        return random.choice(self.genre_bucket[cat])
        return None


def sanitize_author(value: str) -> str:
    if not value: return "Unknown Author"
    lines = [ln.strip() for ln in str(value).splitlines() if ln.strip()]
    first = lines[0] if lines else str(value).strip()
    markers = ("This version is considered", "Project Gutenberg", "www.gutenberg.org")
    for mk in markers:
        p = first.find(mk)
        if p > 0:
            first = first[:p].strip()
            break
    first = " ".join(first.split()).strip()
    return first if first else "Unknown Author"


# =========================================================
# 5. PDF 처리 유틸리티 함수들
# =========================================================
def has_korean(text):
    return bool(re.search(r'[가-힣]', text))


def is_likely_toc_page(text):
    score = 0
    lower_text = text.lower()
    if "contents" in lower_text or "index" in lower_text: score += 2
    if "목차" in text or "차례" in text: score += 5
    if len(re.findall(r'\.{3,}\s*\d+', text)) > 2: score += 5
    chapter_count = len(re.findall(r'(?i)^\s*(chapter|part|제|section)\s*[\dIVXOne]+', text, re.MULTILINE))
    if chapter_count > 4: score += 5
    return score >= 3


def find_start_page_and_author(doc):
    detected_author = "Unknown Author"
    start_page_idx = 0
    if doc.metadata and doc.metadata.get('author') and doc.metadata['author'].strip():
        detected_author = doc.metadata['author']

    for i in range(min(30, len(doc))):
        text = doc[i].get_text()
        if detected_author == "Unknown Author":
            match = re.search(r"(?i)(?:By|Author[:\s]+)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)", text)
            if match: detected_author = match.group(1)
            match_kr = re.search(r"(?:지은이|저자|글)[:\s]+([가-힣]{2,4})", text)
            if match_kr: detected_author = match_kr.group(1)
        if is_likely_toc_page(text): continue
        start_pattern = r"(?i)^\s*((Chapter|Part)\s+(one|1|I)|제?\s*\d+\s*장|프롤로그|서문|머리말|Prologue)\b"
        if re.search(start_pattern, text, re.MULTILINE):
            start_page_idx = i
            break
    return sanitize_author(detected_author), start_page_idx


def analyze_font_characteristics(doc):
    font_data = {}
    for i in range(min(30, len(doc))):
        page = doc[i]
        try:
            blocks = page.get_text("dict")["blocks"]
            for b in blocks:
                if "lines" in b:
                    for line in b["lines"]:
                        for span in line["spans"]:
                            size = round(span["size"])
                            if size < 9: continue
                            is_bold = (span["flags"] & 4) != 0
                            key = (size, is_bold)
                            char_count = len(span["text"])
                            font_data[key] = font_data.get(key, 0) + char_count
        except:
            continue
    if not font_data: return {"body_size": 11, "body_is_bold": False}
    body_key = max(font_data, key=font_data.get)
    return {"body_size": body_key[0], "body_is_bold": body_key[1]}


def extract_chapters_by_font_size(doc, font_info, start_page=0):
    chapters = []
    current_title = "Intro"
    current_content = []
    body_size = font_info["body_size"]
    body_is_bold = font_info["body_is_bold"]
    SIZE_THRESHOLD = body_size + 1.5

    for page_idx in range(start_page, len(doc)):
        page = doc[page_idx]
        blocks = page.get_text("dict")["blocks"]
        blocks.sort(key=lambda b: b["bbox"][1])

        for b in blocks:
            if "lines" in b:
                for line in b["lines"]:
                    for span in line["spans"]:
                        text = span["text"].strip()
                        size = span["size"]
                        is_bold = (span["flags"] & 4) != 0
                        if not text: continue
                        if re.match(r'^\d+$', text) or re.match(r'^\s*(Page|쪽|-)\s*\d+\s*-?$', text,
                                                                re.IGNORECASE): continue

                        is_header = False
                        if size > SIZE_THRESHOLD:
                            is_header = True
                        elif not body_is_bold and is_bold and size >= body_size - 0.5:
                            is_header = True
                        elif body_is_bold and is_bold and size > body_size + 0.5:
                            is_header = True

                        if is_header:
                            if len(text) > 80 or text.endswith('.') or (
                                    not has_korean(text) and text[0].islower()): is_header = False

                        if is_header:
                            if current_content:
                                full_chapter_text = " ".join(current_content)
                                if len(full_chapter_text) > 50:
                                    chapters.append({"title": current_title, "text": full_chapter_text})
                                    current_content = []
                            current_title = text
                        else:
                            current_content.append(
                                text + (" " if has_korean(text) else " " if not text.endswith('-') else ""))
    if current_content:
        chapters.append({"title": current_title, "text": " ".join(current_content)})
    return chapters


def split_into_full_pages(text, words_per_page=300):
    text = re.sub(r"http\S+|www\.\S+", "", text)
    sentences = re.split(r'(?<=[.?!]["\'"])\s+|(?<=[.?!])\s+', text)
    pages = []
    current_page_sentences = []
    current_word_count = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence: continue
        word_count = len(sentence.split())
        if current_word_count + word_count > words_per_page:
            pages.append(" ".join(current_page_sentences))
            current_page_sentences = [sentence]
            current_word_count = word_count
        else:
            current_page_sentences.append(sentence)
            current_word_count += word_count
    if current_page_sentences: pages.append(" ".join(current_page_sentences))
    return pages if pages else ["내용이 없습니다."]


# =========================================================
# 6. 메인 처리 함수 (background_music_job 호출용)
# =========================================================
def process_full_book_for_offline(pdf_path, book_root_folder, music_folder, web_path_prefix):
    """
    [Folder Structure Aware]
    - book_root_folder: users/{id}/{book} -> 책 데이터(JSON, PNG) 저장
    - music_folder: (인자로는 받지만, 음악 파일 저장에는 사용하지 않음)
    - JSON Output: music_path = "music/{filename}" -> 클라이언트가 공용 music 폴더 참조
    """
    filename_base = os.path.splitext(os.path.basename(pdf_path))[0]

    # Mapper 초기화
    mapper = MusicMapper()

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"[Analyzer] PDF 열기 실패: {e}")
        raise e

    # 1. 표지 추출 -> book_root_folder(책 폴더)에 바로 저장
    cover_filename = "default.png"
    try:
        if len(doc) > 0:
            page = doc[0]
            pix = page.get_pixmap()
            cover_save_name = f"{filename_base}.png"
            cover_path = os.path.join(book_root_folder, cover_save_name)
            pix.save(cover_path)
            cover_filename = cover_save_name
    except:
        pass

    # 2. 본문 분석
    real_author, start_page = find_start_page_and_author(doc)
    real_author = sanitize_author(real_author)
    font_info = analyze_font_characteristics(doc)
    raw_chapters_data = extract_chapters_by_font_size(doc, font_info, start_page)
    final_chapters = []

    print(f"[Analyzer] {filename_base} 처리 시작 - 총 {len(raw_chapters_data)} 챕터")

    for ch_idx, ch_data in enumerate(raw_chapters_data):
        chapter_title = ch_data['title']
        chapter_text = ch_data['text']
        chapter_pages = split_into_full_pages(chapter_text, words_per_page=300)

        chapter_segments = []
        pages_buffer = []
        segment_idx = 0

        for p_idx, p_text in enumerate(chapter_pages):
            pages_buffer.append({
                "page_index": p_idx % 3,
                "text": p_text,
                "is_new_segment": (len(pages_buffer) == 0)
            })

            # 세그먼트 생성
            if len(pages_buffer) >= 3 or p_idx == len(chapter_pages) - 1:
                combined_text = " ".join([p['text'] for p in pages_buffer])
                vibe = mapper.analyze_vibe(combined_text)

                # Preset 음악 매칭 시도
                music_info = mapper.get_music(vibe)

                music_filename = None
                music_bpm = 0
                final_music_path = None
                music_source = "ai_generate"

                if music_info:
                    music_filename = music_info['filename']
                    music_bpm = music_info['bpm']

                    # [핵심 경로 설정]
                    # 서버/클라이언트 모두 "music/" 접두사를 보면 "공용 음악 폴더"를 참조함.
                    # 책 폴더 내부에 music 폴더를 만들지 않음.
                    final_music_path = f"music/{music_filename}"
                    music_source = "preset" if not music_info.get("is_custom") else "ai_reused"

                chapter_segments.append({
                    "segment_index": segment_idx,
                    "emotion": vibe,
                    "music_filename": music_filename,  # None이면 BG Job이 AI 생성 후 채움
                    "music_path": final_music_path,
                    "music_source": music_source,
                    "bpm": music_bpm,
                    "pages": pages_buffer
                })
                pages_buffer = []
                segment_idx += 1

        final_chapters.append({
            "chapter_index": ch_idx + 1,
            "title": chapter_title,
            "segments": chapter_segments
        })

    clean_web_prefix = web_path_prefix.rstrip('/')

    book_data = {
        "book_info": {
            "title": filename_base,
            "author": real_author,
            "cover_path": f"{clean_web_prefix}/{cover_filename}",
            "total_chapters": len(final_chapters)
        },
        "chapters": final_chapters
    }

    full_json_filename = f"{filename_base}_full.json"
    json_path = os.path.join(book_root_folder, full_json_filename)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(book_data, f, ensure_ascii=False, indent=2)

    doc.close()

    return {
        "success": True,
        "text_file": full_json_filename,
        "cover_image": cover_filename,
        "real_author": real_author,
        "title": filename_base
    }