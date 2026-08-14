package com.ebookstudio.server.book;

import com.ebookstudio.server.auth.JwtPrincipal;
import com.fasterxml.jackson.annotation.JsonProperty;
import org.springframework.core.io.Resource;
import org.springframework.http.*;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.util.List;
import java.util.Map;

@RestController
public class BookController {
    private final BookService books;

    public BookController(BookService books) {
        this.books = books;
    }

    @PostMapping(path = "/upload_book", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public ResponseEntity<BookService.UploadAccepted> upload(
            @AuthenticationPrincipal JwtPrincipal principal,
            @RequestPart("file") MultipartFile file,
            @RequestPart(value = "request_id", required = false) String requestId) {
        return ResponseEntity.accepted().body(books.upload(principal, file, requestId));
    }

    @GetMapping("/check_status/{jobId}")
    public BookService.JobStatus status(@AuthenticationPrincipal JwtPrincipal principal,
                                        @PathVariable String jobId) {
        return books.status(principal, jobId);
    }

    @DeleteMapping("/jobs/{jobId}")
    public BookService.JobStatus cancel(@AuthenticationPrincipal JwtPrincipal principal,
                                        @PathVariable String jobId) {
        return books.cancel(principal, jobId);
    }

    @PostMapping("/my_books")
    public Map<String, List<BookService.BookSummary>> myBooks(@AuthenticationPrincipal JwtPrincipal principal) {
        return Map.of("books", books.myBooks(principal));
    }

    @PostMapping("/delete_server_book")
    public Map<String, String> delete(@AuthenticationPrincipal JwtPrincipal principal,
                                      @RequestBody DeleteBookRequest request) {
        books.deleteBook(principal, request.bookFolder());
        return Map.of("message", "Deleted successfully");
    }

    @GetMapping("/list_music_files/{username}/{bookFolder}")
    public Map<String, List<String>> listMusic(@AuthenticationPrincipal JwtPrincipal principal,
                                               @PathVariable String username,
                                               @PathVariable String bookFolder) {
        return Map.of("files", books.listMusic(principal, username, bookFolder));
    }

    @GetMapping("/files/{username}/{bookFolder}/{filename}")
    public ResponseEntity<Resource> rootFile(@AuthenticationPrincipal JwtPrincipal principal,
                                             @PathVariable String username,
                                             @PathVariable String bookFolder,
                                             @PathVariable String filename) {
        return resourceResponse(books.rootFile(principal, username, bookFolder, filename));
    }

    @GetMapping("/files/{username}/{bookFolder}/music/{filename}")
    public ResponseEntity<Resource> musicFile(@AuthenticationPrincipal JwtPrincipal principal,
                                              @PathVariable String username,
                                              @PathVariable String bookFolder,
                                              @PathVariable String filename) {
        return resourceResponse(books.musicFile(principal, username, bookFolder, filename));
    }

    private static ResponseEntity<Resource> resourceResponse(Resource resource) {
        MediaType type = MediaTypeFactory.getMediaType(resource).orElse(MediaType.APPLICATION_OCTET_STREAM);
        return ResponseEntity.ok().contentType(type)
                .header(HttpHeaders.CONTENT_DISPOSITION,
                        ContentDisposition.inline().filename(resource.getFilename()).build().toString())
                .body(resource);
    }

    public record DeleteBookRequest(@JsonProperty("book_folder") String bookFolder) { }
}
