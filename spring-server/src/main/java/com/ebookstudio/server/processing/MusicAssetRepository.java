package com.ebookstudio.server.processing;

import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Repository;

import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.Optional;

@Repository
public class MusicAssetRepository {
    private final JdbcTemplate jdbc;
    private final boolean postgresql;

    public MusicAssetRepository(JdbcTemplate jdbc) {
        this.jdbc = jdbc;
        this.postgresql = Boolean.TRUE.equals(jdbc.execute((org.springframework.jdbc.core.ConnectionCallback<Boolean>) connection ->
                "PostgreSQL".equalsIgnoreCase(
                        connection.getMetaData().getDatabaseProductName())));
    }

    public MusicAsset insert(MusicAsset asset) {
        if (asset.id() != null) throw new IllegalArgumentException("New music asset cannot have an id");
        jdbc.update("""
                INSERT INTO music_assets(
                    signature, asset_source, prompt, genre, bpm, keywords_json,
                    model_name, model_version, generation_params_json, storage_key,
                    checksum, duration_seconds, status, reuse_count,
                    created_at, updated_at, last_used_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, asset.signature(), asset.assetSource().name(), asset.prompt(),
                asset.genre(), asset.bpm(), asset.keywordsJson(), asset.modelName(),
                asset.modelVersion(), asset.generationParamsJson(), asset.storageKey(),
                asset.checksum(), asset.durationSeconds(), asset.status().name(),
                asset.reuseCount(), asset.createdAt(), asset.updatedAt(), asset.lastUsedAt());
        return findBySignature(asset.signature()).orElseThrow();
    }

    public synchronized MusicAsset saveSnapshot(MusicAsset asset) {
        if (asset.id() != null) throw new IllegalArgumentException("Music snapshot cannot have an id");
        if (postgresql) {
            insertSnapshot(asset, true);
        } else if (findBySignature(asset.signature()).isEmpty()) {
            insertSnapshot(asset, false);
        }
        jdbc.update("""
                UPDATE music_assets
                SET asset_source=?, prompt=?, genre=?, bpm=?, keywords_json=?,
                    model_name=?, model_version=?, generation_params_json=?, storage_key=?,
                    checksum=?, duration_seconds=?, status=?,
                    reuse_count=CASE WHEN reuse_count > ? THEN reuse_count ELSE ? END,
                    updated_at=?, last_used_at=COALESCE(?, last_used_at)
                WHERE signature=?
                """, asset.assetSource().name(), asset.prompt(), asset.genre(), asset.bpm(),
                asset.keywordsJson(), asset.modelName(), asset.modelVersion(),
                asset.generationParamsJson(), asset.storageKey(), asset.checksum(),
                asset.durationSeconds(), asset.status().name(), asset.reuseCount(),
                asset.reuseCount(), asset.updatedAt(), asset.lastUsedAt(), asset.signature());
        return findBySignature(asset.signature()).orElseThrow();
    }

    private void insertSnapshot(MusicAsset asset, boolean ignoreSignatureConflict) {
        String conflictClause = ignoreSignatureConflict
                ? " ON CONFLICT(signature) DO NOTHING" : "";
        jdbc.update("""
                INSERT INTO music_assets(
                    signature, asset_source, prompt, genre, bpm, keywords_json,
                    model_name, model_version, generation_params_json, storage_key,
                    checksum, duration_seconds, status, reuse_count,
                    created_at, updated_at, last_used_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """ + conflictClause, asset.signature(), asset.assetSource().name(),
                asset.prompt(), asset.genre(), asset.bpm(), asset.keywordsJson(),
                asset.modelName(), asset.modelVersion(), asset.generationParamsJson(),
                asset.storageKey(), asset.checksum(), asset.durationSeconds(),
                asset.status().name(), asset.reuseCount(), asset.createdAt(),
                asset.updatedAt(), asset.lastUsedAt());
    }

    public Optional<MusicAsset> findById(long id) {
        return jdbc.query(selectSql() + " WHERE id=?", rs ->
                rs.next() ? Optional.of(map(rs, "")) : Optional.empty(), id);
    }

    public Optional<MusicAsset> findBySignature(String signature) {
        return jdbc.query(selectSql() + " WHERE signature=?", rs ->
                rs.next() ? Optional.of(map(rs, "")) : Optional.empty(), signature);
    }

    public boolean markReady(long id, String storageKey, String checksum,
                             int durationSeconds, long now) {
        return jdbc.update("""
                UPDATE music_assets
                SET storage_key=?, checksum=?, duration_seconds=?, status='READY',
                    updated_at=?, last_used_at=?
                WHERE id=? AND status='GENERATING'
                """, storageKey, checksum, durationSeconds, now, now, id) == 1;
    }

    public boolean markFailed(long id, long now) {
        return jdbc.update("""
                UPDATE music_assets SET status='FAILED', updated_at=?
                WHERE id=? AND status='GENERATING'
                """, now, id) == 1;
    }

    public boolean recordReuse(long id, long now) {
        return jdbc.update("""
                UPDATE music_assets
                SET reuse_count=reuse_count+1, last_used_at=?, updated_at=?
                WHERE id=? AND status='READY'
                """, now, now, id) == 1;
    }

    static MusicAsset map(ResultSet rs, String prefix) throws SQLException {
        int duration = rs.getInt(prefix + "duration_seconds");
        Integer nullableDuration = rs.wasNull() ? null : duration;
        long lastUsed = rs.getLong(prefix + "last_used_at");
        Long nullableLastUsed = rs.wasNull() ? null : lastUsed;
        return new MusicAsset(
                rs.getLong(prefix + "id"), rs.getString(prefix + "signature"),
                MusicAsset.AssetSource.valueOf(rs.getString(prefix + "asset_source")),
                rs.getString(prefix + "prompt"), rs.getString(prefix + "genre"),
                rs.getInt(prefix + "bpm"), rs.getString(prefix + "keywords_json"),
                rs.getString(prefix + "model_name"), rs.getString(prefix + "model_version"),
                rs.getString(prefix + "generation_params_json"), rs.getString(prefix + "storage_key"),
                rs.getString(prefix + "checksum"), nullableDuration,
                MusicAsset.Status.valueOf(rs.getString(prefix + "status")),
                rs.getInt(prefix + "reuse_count"), rs.getLong(prefix + "created_at"),
                rs.getLong(prefix + "updated_at"), nullableLastUsed);
    }

    private static String selectSql() {
        return """
                SELECT id, signature, asset_source, prompt, genre, bpm, keywords_json,
                       model_name, model_version, generation_params_json, storage_key,
                       checksum, duration_seconds, status, reuse_count,
                       created_at, updated_at, last_used_at
                FROM music_assets
                """;
    }
}
