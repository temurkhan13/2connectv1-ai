"""PostgreSQL adapter with pgvector for embedding storage."""
import logging
import os
from typing import List, Dict, Any, Optional
import psycopg2
from psycopg2.extras import RealDictCursor, Json
import pgvector.psycopg2
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class PostgreSQLAdapter:
    """PostgreSQL adapter for embedding storage with pgvector."""

    def __init__(self):
        """Initialize PostgreSQL connection."""
        self.database_url = os.getenv('DATABASE_URL')
        if not self.database_url:
            raise ValueError("DATABASE_URL environment variable is required")

        # Backend database URL for operations on users/matches (port 5432)
        # This is separate from the AI database (port 5433) which stores embeddings
        self.backend_database_url = os.getenv('RECIPROCITY_BACKEND_DB_URL')
        if not self.backend_database_url:
            logger.warning("RECIPROCITY_BACKEND_DB_URL not set - user status updates may fail")

        # Register pgvector extension (properly close the registration connection)
        registration_conn = psycopg2.connect(self.database_url)
        try:
            pgvector.psycopg2.register_vector(registration_conn)
        finally:
            registration_conn.close()

    def get_connection(self):
        """Get database connection to AI database. Caller is responsible for closing."""
        return psycopg2.connect(self.database_url)

    def get_backend_connection(self):
        """Get database connection to backend database (for user/match operations).

        The backend database (port 5432) stores users, matches, profiles etc.
        The AI database (port 5433) stores embeddings and vectors.
        """
        if not self.backend_database_url:
            raise ValueError("RECIPROCITY_BACKEND_DB_URL environment variable is required for user operations")
        return psycopg2.connect(self.backend_database_url)
    
    def store_embedding(self,
                       user_id: str,
                       embedding_type: str,
                       vector_data: List[float],
                       metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Store embedding vector in PostgreSQL."""
        conn = None
        cursor = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            # Convert numpy arrays to plain Python floats for PostgreSQL vector format
            if hasattr(vector_data, 'tolist'):
                # numpy array
                vector_list = vector_data.tolist()
            else:
                # Ensure all values are plain Python floats (not np.float32)
                vector_list = [float(v) for v in vector_data]

            # Upsert embedding (insert or update if exists)
            query = """
                INSERT INTO user_embeddings (user_id, embedding_type, vector_data, metadata, created_at, updated_at)
                VALUES (%s, %s, %s, %s, (CURRENT_TIMESTAMP AT TIME ZONE 'UTC'), (CURRENT_TIMESTAMP AT TIME ZONE 'UTC'))
                ON CONFLICT (user_id, embedding_type)
                DO UPDATE SET
                    vector_data = EXCLUDED.vector_data,
                    metadata = EXCLUDED.metadata,
                    updated_at = (CURRENT_TIMESTAMP AT TIME ZONE 'UTC')
            """

            cursor.execute(query, (
                user_id,
                embedding_type,
                vector_list,
                Json(metadata or {})
            ))

            conn.commit()
            logger.info(f"Stored {embedding_type} embedding for user {user_id} ({len(vector_data)} dimensions)")
            return True

        except Exception as e:
            logger.error(f"Error storing embedding for user {user_id}: {str(e)}")
            return False
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()
    
    def get_user_embeddings(self, user_id: str) -> Dict[str, Optional[Dict[str, Any]]]:
        """
        Get all embeddings for a user.

        Returns ALL embedding types including multi-vector embeddings like:
        - requirements, offerings (basic 2-vector)
        - requirements_industry, offerings_industry, etc. (multi-vector)
        """
        conn = None
        cursor = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            query = """
                SELECT embedding_type, vector_data, metadata, created_at, updated_at
                FROM user_embeddings
                WHERE user_id = %s
            """

            cursor.execute(query, (user_id,))
            results = cursor.fetchall()

            # Return ALL embedding types, not just requirements/offerings
            embeddings = {}
            for result in results:
                embedding_type = result['embedding_type']
                embeddings[embedding_type] = dict(result)

            return embeddings

        except Exception as e:
            logger.error(f"Error retrieving embeddings for user {user_id}: {str(e)}")
            return {}
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()
    
    def get_all_user_embeddings(self, embedding_type: str = None) -> List[Dict[str, Any]]:
        """Get all user embeddings, optionally filtered by type."""
        conn = None
        cursor = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            if embedding_type:
                query = """
                    SELECT user_id, embedding_type, vector_data, metadata, created_at
                    FROM user_embeddings
                    WHERE embedding_type = %s
                    ORDER BY created_at DESC
                """
                cursor.execute(query, (embedding_type,))
            else:
                query = """
                    SELECT user_id, embedding_type, vector_data, metadata, created_at
                    FROM user_embeddings
                    ORDER BY created_at DESC
                """
                cursor.execute(query)

            results = cursor.fetchall()
            return [dict(result) for result in results]

        except Exception as e:
            logger.error(f"Error retrieving all embeddings: {str(e)}")
            return []
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()
    
    def find_similar_users(self,
                          query_vector: List[float],
                          embedding_type: str,
                          threshold: float = 0.7,
                          upper_threshold: float = None,
                          exclude_user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Find users with similar embeddings using pgvector cosine similarity."""
        conn = None
        cursor = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            # Use pgvector cosine similarity with upper bound filter
            if exclude_user_id:
                if upper_threshold is not None:
                    # With upper bound - filter out too-similar matches (false partial matches)
                    query = (
                        "SELECT user_id, (1 - (vector_data <=> %s)) as similarity_score, metadata "
                        "FROM user_embeddings "
                        "WHERE embedding_type = %s "
                        "AND user_id != %s "
                        "AND (1 - (vector_data <=> %s)) >= %s "
                        "AND (1 - (vector_data <=> %s)) <= %s "
                        "ORDER BY vector_data <=> %s"
                    )
                    cursor.execute(query, (query_vector, embedding_type, exclude_user_id, query_vector, threshold, query_vector, upper_threshold, query_vector))
                else:
                    # Without upper bound (original behavior)
                    query = (
                        "SELECT user_id, (1 - (vector_data <=> %s)) as similarity_score, metadata "
                        "FROM user_embeddings "
                        "WHERE embedding_type = %s "
                        "AND user_id != %s "
                        "AND (1 - (vector_data <=> %s)) >= %s "
                        "ORDER BY vector_data <=> %s"
                    )
                    cursor.execute(query, (query_vector, embedding_type, exclude_user_id, query_vector, threshold, query_vector))
            else:
                if upper_threshold is not None:
                    # With upper bound
                    query = (
                        "SELECT user_id, (1 - (vector_data <=> %s)) as similarity_score, metadata "
                        "FROM user_embeddings "
                        "WHERE embedding_type = %s "
                        "AND (1 - (vector_data <=> %s)) >= %s "
                        "AND (1 - (vector_data <=> %s)) <= %s "
                        "ORDER BY vector_data <=> %s"
                    )
                    cursor.execute(query, (query_vector, embedding_type, query_vector, threshold, query_vector, upper_threshold, query_vector))
                else:
                    # Without upper bound (original behavior)
                    query = (
                        "SELECT user_id, (1 - (vector_data <=> %s)) as similarity_score, metadata "
                        "FROM user_embeddings "
                        "WHERE embedding_type = %s "
                        "AND (1 - (vector_data <=> %s)) >= %s "
                        "ORDER BY vector_data <=> %s"
                    )
                    cursor.execute(query, (query_vector, embedding_type, query_vector, threshold, query_vector))

            results = cursor.fetchall()
            return [dict(result) for result in results]

        except Exception as e:
            logger.error(f"Error finding similar users: {str(e)}")
            return []
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()

    def find_multi_vector_matches(self,
                                 user_id: str,
                                 dimension_weights: Dict[str, float],
                                 limit: int = 50) -> List[Dict[str, Any]]:
        """
        Find best matches using weighted multi-vector similarity in SQL.
        
        Optimized to perform the weighted sum of cosine similarities directly in the database.
        This avoids loading all user vectors into memory.
        
        Args:
            user_id: The ID of the user to find matches for (The "Viewer")
            dimension_weights: Dict of {dimension_name: weight}
            limit: Max users to return
            
        Returns:
            List of match candidates with total_score and dimension breakdown
        """
        conn = None
        cursor = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            # 1. Fetch Viewer's Requirement Embeddings first
            # We need these vectors to construct the similarity query
            cursor.execute("""
                SELECT embedding_type, vector_data 
                FROM user_embeddings 
                WHERE user_id = %s AND embedding_type LIKE 'requirements_%%'
            """, (user_id,))
            
            viewer_rows = cursor.fetchall()
            if not viewer_rows:
                return []
                
            # Map dimension -> vector string for SQL
            # embedding_type format is "requirements_{dimension}"
            viewer_vectors = {}
            for row in viewer_rows:
                dim = row['embedding_type'].replace('requirements_', '')
                # Ensure vector is formatted for pgvector (string representation)
                # CRITICAL: Convert numpy floats to Python floats before string conversion
                vec_data = row['vector_data']
                if hasattr(vec_data, 'tolist'):
                    # numpy array - tolist() converts to native Python types
                    vec_list = vec_data.tolist()
                else:
                    # Could be list with numpy floats - ensure all are Python floats
                    vec_list = [float(v) for v in vec_data]
                vec_str = str(vec_list)
                viewer_vectors[dim] = vec_str

            # 2. Build Dynamic SQL Query
            # We want to sum the weighted cosine similarity for each dimension.
            # Cosine similarity in pgvector is (1 - (a <=> b)).
            
            # Subquery to pivot offerings for all other users
            # We join user_embeddings on itself or aggregate? 
            # Better approach: 
            # Select user_id, 
            # SUM(CASE WHEN type='offerings_industry' THEN (1 - (vec <=> viewer_ind_vec)) * weight_ind ... END) as total_score
            
            select_clauses = []
            params = []
            
            # Only include dimensions where the viewer actually has a requirement vector
            active_dimensions = []
            
            for dim, weight in dimension_weights.items():
                if dim in viewer_vectors:
                    # (1 - (vector_data <=> %s)) * %s
                    # We cast to float just in case
                    key = f"offerings_{dim}"
                    # We use a CASE statement inside SUM to handle missing dimensions for candidates (treat as 0 similarity)
                    # Actually, we can just filter by the relevant types
                    active_dimensions.append(dim)
            
            if not active_dimensions:
                return []

            # We need to compute score per user.
            # A user might have multiple rows (one per dimension).
            # We group by user_id.
            
            # Query Structure:
            # SELECT user_id, SUM(weighted_score) as total_score, array_agg(json_build_object('dim', dim, 'score', raw_score)) as details
            # FROM (
            #   SELECT user_id, 
            #     CASE 
            #       WHEN embedding_type = 'offerings_industry' THEN (1 - (vector_data <=> %s)) * %s
            #       ...
            #     END as weighted_score,
            #     ...
            #   FROM user_embeddings
            #   WHERE user_id != %s
            #   AND embedding_type IN (...)
            # ) scores
            # GROUP BY user_id
            # ORDER BY total_score DESC
            
            query_parts = []
            query_params = []
            
            type_filters = []
            
            for dim in active_dimensions:
                emb_type = f"offerings_{dim}"
                vec_str = viewer_vectors[dim]
                weight = dimension_weights[dim]
                
                # WHEN embedding_type = 'offerings_industry' THEN (1 - (vector_data <=> '[...]')) * 0.25
                part = f"WHEN embedding_type = %s THEN (1 - (vector_data <=> %s)) * %s"
                query_parts.append(part)
                query_params.extend([emb_type, vec_str, weight])
                
                type_filters.append(emb_type)

            if not query_parts:
                return []

            full_query = f"""
                SELECT 
                    user_id,
                    SUM(match_score) as total_score,
                    json_object_agg(dimension, raw_similarity) as dimension_scores
                FROM (
                    SELECT 
                        user_id,
                        REPLACE(embedding_type, 'offerings_', '') as dimension,
                        CASE 
                            {' '.join(query_parts)}
                            ELSE 0 
                        END as match_score,
                        CASE
                            {' '.join([p.replace(f" * %s", "") for p in query_parts])} -- Recalculate raw sim without weight for display
                            ELSE 0
                        END as raw_similarity
                    FROM user_embeddings
                    WHERE user_id != %s
                    AND embedding_type = ANY(%s)
                ) calculated
                GROUP BY user_id
                ORDER BY total_score DESC
                LIMIT %s
            """
            
            # Add final params
            # Note: inside query_parts we added [type, vec, weight] triples
            # For raw_similarity recreation, we need [type, vec] pairs. 
            # This is duplicate parameter passing. A cleaner way with CTEs might be better but this is fine for now.
            
            # Actually, let's simplify. We can calculate raw similarity and just multiply by weight in outer query if we map it.
            # But grouping is easier if we just calc it inside.
            
            # Re-assembling params carefully:
            # 1. Calc Parts: [type, vec, weight] * N
            # 2. Raw Parts: [type, vec] * N
            # 3. Where: user_id
            # 4. Where: type_list
            # 5. Limit
            
            final_params = []
            final_params.extend(query_params) # [type, vec, weight]...
            
            # Add params for raw_similarity CASE statement (same logic minus weight)
            for dim in active_dimensions:
                emb_type = f"offerings_{dim}"
                vec_str = viewer_vectors[dim]
                final_params.extend([emb_type, vec_str])
            
            final_params.append(user_id)
            final_params.append(type_filters)
            final_params.append(limit)

            cursor.execute(full_query, final_params)
            results = cursor.fetchall()
            
            return [dict(row) for row in results]

        except Exception as e:
            logger.error(f"Error finding multi-vector matches in DB: {str(e)}")
            return []
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()
    
    def health_check(self) -> bool:
        """Check if PostgreSQL is accessible."""
        conn = None
        cursor = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            return result[0] == 1

        except Exception as e:
            logger.error(f"Database health check failed: {str(e)}")
            return False
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()

    def get_embedding_stats(self) -> Dict[str, Any]:
        """Get statistics about users and embeddings for system verification."""
        conn = None
        cursor = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            # Get total user count
            cursor.execute("SELECT COUNT(*) FROM users")
            total_users = cursor.fetchone()[0]

            # Get count of users with embeddings (distinct user_ids in user_embeddings)
            cursor.execute("SELECT COUNT(DISTINCT user_id) FROM user_embeddings")
            users_with_embeddings = cursor.fetchone()[0]

            # Get total embedding count
            cursor.execute("SELECT COUNT(*) FROM user_embeddings")
            total_embeddings = cursor.fetchone()[0]

            return {
                "total_users": total_users,
                "users_with_embeddings": users_with_embeddings,
                "total_embeddings": total_embeddings
            }

        except Exception as e:
            logger.error(f"Error getting embedding stats: {str(e)}")
            return {
                "total_users": 0,
                "users_with_embeddings": 0,
                "total_embeddings": 0,
                "error": str(e)
            }
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()


    def update_user_onboarding_status(self, user_id: str, status: str = 'completed') -> bool:
        """
        Update user's onboarding_status in the users table.

        This bridges the AI service onboarding completion to the backend's user status,
        allowing users to access the dashboard after completing onboarding via the AI service.

        IMPORTANT: Uses backend database connection (port 5432) because the users table
        is in the backend DB (reciprocity_db), not the AI DB (reciprocity_ai on port 5433).

        Args:
            user_id: The user's UUID
            status: The onboarding status to set (default: 'completed')

        Returns:
            True if updated successfully, False otherwise
        """
        conn = None
        cursor = None
        try:
            conn = self.get_backend_connection()
            cursor = conn.cursor()

            query = """
                UPDATE users
                SET onboarding_status = %s, updated_at = (CURRENT_TIMESTAMP AT TIME ZONE 'UTC')
                WHERE id = %s
            """
            cursor.execute(query, (status, user_id))

            rows_affected = cursor.rowcount
            conn.commit()

            if rows_affected > 0:
                logger.info(f"Updated onboarding_status to '{status}' for user {user_id}")
                return True
            else:
                logger.warning(f"No user found with id {user_id} to update onboarding_status")
                return False

        except Exception as e:
            logger.error(f"Error updating onboarding_status for user {user_id}: {str(e)}")
            return False
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()

    def create_user_summary(
        self,
        user_id: str,
        summary: str,
        status: str = 'draft',
        urgency: str = 'ongoing'
    ) -> Optional[str]:
        """
        Create a user summary record for the Discover page.

        This is called during onboarding completion to ensure users appear
        in the Discover page search results. The summary contains the user's
        profile information in a searchable format.

        IMPORTANT: Uses backend database connection (port 5432) because user_summaries
        is in the backend DB (reciprocity_db), not the AI DB (reciprocity_ai on port 5433).

        Args:
            user_id: The user's UUID
            summary: JSON string of the user's profile summary
            status: Summary status (draft, approved, etc.)
            urgency: User's urgency level

        Returns:
            The created summary UUID, or None if failed
        """
        import uuid
        conn = None
        cursor = None
        try:
            conn = self.get_backend_connection()
            cursor = conn.cursor()

            # Get the latest version for this user
            version_query = """
                SELECT COALESCE(MAX(version), 0) + 1 FROM user_summaries WHERE user_id = %s
            """
            cursor.execute(version_query, (user_id,))
            next_version = cursor.fetchone()[0]

            # Generate UUID for the summary
            summary_id = str(uuid.uuid4())

            # Insert the summary (table allows multiple versions per user)
            insert_query = """
                INSERT INTO user_summaries (id, user_id, status, summary, version, webhook, urgency, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, true, %s, (CURRENT_TIMESTAMP AT TIME ZONE 'UTC'), (CURRENT_TIMESTAMP AT TIME ZONE 'UTC'))
                RETURNING id
            """
            cursor.execute(insert_query, (summary_id, user_id, status, summary, next_version, urgency))
            result = cursor.fetchone()
            conn.commit()

            if result:
                logger.info(f"Created user_summary for user {user_id} (version {next_version})")
                return result[0]
            else:
                logger.warning(f"Failed to create user_summary for user {user_id}")
                return None

        except Exception as e:
            logger.error(f"Error creating user_summary for user {user_id}: {str(e)}")
            if conn:
                conn.rollback()
            return None
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()


# Global adapter instance
postgresql_adapter = PostgreSQLAdapter()
