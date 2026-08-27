package com.ebookstudio.server.processing;

import com.ebookstudio.server.auth.JwtPrincipal;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/books")
public class BookInsightsController {
    private final BookInsightsService insights;

    public BookInsightsController(BookInsightsService insights) {
        this.insights = insights;
    }

    @GetMapping("/{bookFolder}/processing-history")
    public BookInsightsService.ProcessingHistoryResponse processingHistory(
            @AuthenticationPrincipal JwtPrincipal principal,
            @PathVariable("bookFolder") String bookFolder) {
        return insights.processingHistory(principal, bookFolder);
    }

    @GetMapping("/{bookFolder}/music-tracks")
    public BookInsightsService.MusicTracksResponse musicTracks(
            @AuthenticationPrincipal JwtPrincipal principal,
            @PathVariable("bookFolder") String bookFolder) {
        return insights.musicTracks(principal, bookFolder);
    }
}
