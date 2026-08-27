package com.ebookstudio.server.processing;

import org.springframework.dao.DuplicateKeyException;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Repository;

import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.List;

@Repository
public class BookMusicBindingRepository {
    private final JdbcTemplate jdbc;

    public BookMusicBindingRepository(JdbcTemplate jdbc) {
        this.jdbc = jdbc;
    }

    public void save(BookMusicBinding binding) {
        int updated = update(binding);
        if (updated == 1) return;
        try {
            jdbc.update("""
                    INSERT INTO book_music_bindings(
                        book_id, segment_key, music_asset_id, processing_run_id,
                        binding_type, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, binding.bookId(), binding.segmentKey(), binding.musicAssetId(),
                    binding.processingRunId(), binding.bindingType().name(),
                    binding.createdAt(), binding.updatedAt());
        } catch (DuplicateKeyException race) {
            if (update(binding) != 1) throw race;
        }
    }

    public List<BookMusicTrack> findTracksByBookId(long bookId) {
        return jdbc.query("""
                SELECT bm.book_id, bm.segment_key, bm.music_asset_id, bm.processing_run_id,
                       bm.binding_type, bm.created_at AS binding_created_at,
                       bm.updated_at AS binding_updated_at,
                       m.id AS music_id, m.signature AS music_signature,
                       m.asset_source AS music_asset_source, m.prompt AS music_prompt,
                       m.genre AS music_genre, m.bpm AS music_bpm,
                       m.keywords_json AS music_keywords_json,
                       m.model_name AS music_model_name,
                       m.model_version AS music_model_version,
                       m.generation_params_json AS music_generation_params_json,
                       m.storage_key AS music_storage_key, m.checksum AS music_checksum,
                       m.duration_seconds AS music_duration_seconds,
                       m.status AS music_status, m.reuse_count AS music_reuse_count,
                       m.created_at AS music_created_at, m.updated_at AS music_updated_at,
                       m.last_used_at AS music_last_used_at
                FROM book_music_bindings bm
                JOIN music_assets m ON m.id=bm.music_asset_id
                WHERE bm.book_id=?
                ORDER BY bm.segment_key
                """, (rs, rowNum) -> new BookMusicTrack(mapBinding(rs),
                MusicAssetRepository.map(rs, "music_")), bookId);
    }

    public int deleteByBookId(long bookId) {
        return jdbc.update("DELETE FROM book_music_bindings WHERE book_id=?", bookId);
    }

    private int update(BookMusicBinding binding) {
        return jdbc.update("""
                UPDATE book_music_bindings
                SET music_asset_id=?, processing_run_id=?, binding_type=?, updated_at=?
                WHERE book_id=? AND segment_key=?
                """, binding.musicAssetId(), binding.processingRunId(),
                binding.bindingType().name(), binding.updatedAt(),
                binding.bookId(), binding.segmentKey());
    }

    private static BookMusicBinding mapBinding(ResultSet rs) throws SQLException {
        long processingRunId = rs.getLong("processing_run_id");
        Long nullableProcessingRunId = rs.wasNull() ? null : processingRunId;
        return new BookMusicBinding(rs.getLong("book_id"), rs.getString("segment_key"),
                rs.getLong("music_asset_id"), nullableProcessingRunId,
                BookMusicBinding.BindingType.valueOf(rs.getString("binding_type")),
                rs.getLong("binding_created_at"), rs.getLong("binding_updated_at"));
    }
}
