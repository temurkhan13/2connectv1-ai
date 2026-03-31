-- Chat tables for Supabase Realtime
-- These mirror the backend Sequelize migration but add RLS for direct client access

-- 1. Chat Conversations
CREATE TABLE IF NOT EXISTS chat_conversations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user1_id UUID NOT NULL,
    user2_id UUID NOT NULL,
    match_id UUID,
    last_message_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(user1_id, user2_id)
);

CREATE INDEX idx_chat_conv_user1 ON chat_conversations(user1_id);
CREATE INDEX idx_chat_conv_user2 ON chat_conversations(user2_id);
CREATE INDEX idx_chat_conv_last_msg ON chat_conversations(last_message_at DESC);

-- 2. Chat Messages
CREATE TABLE IF NOT EXISTS chat_messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID NOT NULL REFERENCES chat_conversations(id) ON DELETE CASCADE,
    sender_id UUID NOT NULL,
    content TEXT NOT NULL,
    message_type TEXT NOT NULL DEFAULT 'text' CHECK (message_type IN ('text', 'image', 'system')),
    read_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_chat_msg_conv_time ON chat_messages(conversation_id, created_at);
CREATE INDEX idx_chat_msg_unread ON chat_messages(conversation_id, sender_id, read_at);

-- 3. Enable RLS
ALTER TABLE chat_conversations ENABLE ROW LEVEL SECURITY;
ALTER TABLE chat_messages ENABLE ROW LEVEL SECURITY;

-- 4. RLS Policies for chat_conversations
-- Users can only see conversations they are part of
CREATE POLICY "Users can view own conversations"
    ON chat_conversations FOR SELECT
    USING (auth.uid() = user1_id OR auth.uid() = user2_id);

CREATE POLICY "Users can create conversations they are part of"
    ON chat_conversations FOR INSERT
    WITH CHECK (auth.uid() = user1_id OR auth.uid() = user2_id);

CREATE POLICY "Users can update own conversations"
    ON chat_conversations FOR UPDATE
    USING (auth.uid() = user1_id OR auth.uid() = user2_id);

-- 5. RLS Policies for chat_messages
-- Users can only see messages in their conversations
CREATE POLICY "Users can view messages in own conversations"
    ON chat_messages FOR SELECT
    USING (
        EXISTS (
            SELECT 1 FROM chat_conversations c
            WHERE c.id = chat_messages.conversation_id
            AND (c.user1_id = auth.uid() OR c.user2_id = auth.uid())
        )
    );

-- Users can send messages in their conversations
CREATE POLICY "Users can send messages in own conversations"
    ON chat_messages FOR INSERT
    WITH CHECK (
        auth.uid() = sender_id
        AND EXISTS (
            SELECT 1 FROM chat_conversations c
            WHERE c.id = chat_messages.conversation_id
            AND (c.user1_id = auth.uid() OR c.user2_id = auth.uid())
        )
    );

-- Users can update read_at on messages sent TO them
CREATE POLICY "Users can mark messages as read"
    ON chat_messages FOR UPDATE
    USING (
        sender_id != auth.uid()
        AND EXISTS (
            SELECT 1 FROM chat_conversations c
            WHERE c.id = chat_messages.conversation_id
            AND (c.user1_id = auth.uid() OR c.user2_id = auth.uid())
        )
    );

-- 6. Enable Realtime for chat_messages (this is what powers live updates)
ALTER PUBLICATION supabase_realtime ADD TABLE chat_messages;

-- 7. Function to auto-update last_message_at on new message
CREATE OR REPLACE FUNCTION update_conversation_last_message()
RETURNS TRIGGER AS $$
BEGIN
    UPDATE chat_conversations
    SET last_message_at = NEW.created_at, updated_at = NEW.created_at
    WHERE id = NEW.conversation_id;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_last_message
    AFTER INSERT ON chat_messages
    FOR EACH ROW
    EXECUTE FUNCTION update_conversation_last_message();

-- 8. Service role policy (for backend API to manage conversations)
-- The backend uses service_role key, which bypasses RLS
-- No additional policy needed for service role access
