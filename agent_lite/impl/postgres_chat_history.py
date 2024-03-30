#
# requires installation of sqlalchemy and asyncpg
from dataclasses import dataclass
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from sqlalchemy import (
        Integer,
        String,
        Text,
        delete,
        select,
        update,
    )
    from sqlalchemy.ext.asyncio import (
        AsyncSession,
        async_sessionmaker,
        create_async_engine,
    )
    from sqlalchemy.orm import DeclarativeBase, mapped_column

from agent_lite.core import BaseChatHistory, Message

if TYPE_CHECKING:

    class Base(DeclarativeBase):
        pass

    class AgentConversationHistory(Base):
        __tablename__ = "agent_conversation_history"
        id = mapped_column(Integer, primary_key=True)
        session_id = mapped_column(String)
        role = mapped_column(String, nullable=False)
        content = mapped_column(Text)

else:
    Base, AgentConversationHistory = None, None


@dataclass
class PostgresChatHistory(BaseChatHistory):
    url: str
    session_id: str

    def __post_init__(self):
        # replace protocol with postgres+asyncpg for asyncpg driver
        async_url = self.url.replace("postgresql://", "postgresql+asyncpg://")
        async_engine = create_async_engine(async_url, pool_pre_ping=True)

        self.db_session_maker = async_sessionmaker(
            async_engine, expire_on_commit=False, class_=AsyncSession
        )

    async def add_message(self, message: Message) -> None:

        new_history_item = AgentConversationHistory(
            session_id=self.session_id,
            role=message.message_type(),
            content=message.serialized_message_content(),
        )
        async with self.db_session_maker() as db_session:
            db_session.add(new_history_item)
            await db_session.commit()

    async def delete_message(self, index: int) -> None:
        async with self.db_session_maker() as db_session:
            message_id = await self._message_id_by_index(db_session, index)
            stmt = delete(AgentConversationHistory).where(
                AgentConversationHistory.id == message_id
            )
            await db_session.execute(stmt)
            await db_session.commit()

    async def edit_message(self, index: int, message: Message) -> None:
        async with self.db_session_maker() as db_session:
            message_id = await self._message_id_by_index(db_session, index)

            stmt = (
                update(AgentConversationHistory)
                .where(AgentConversationHistory.id == message_id)
                .values(
                    role=message.message_type(),
                    content=message.serialized_message_content(),
                )
            )
            await db_session.execute(stmt)
            await db_session.commit()

    async def get_messages(self) -> List[Message]:
        async with self.db_session_maker() as db_session:
            all_messages = await self._get_all_messages(db_session)
            return [
                Message.from_serialized(
                    message_type=m.role, serialized_content=m.content
                )
                for m in all_messages
            ]

    async def delete_range(self, start: int, end: int) -> None:
        async with self.db_session_maker() as db_session:
            all_messages = await self._get_all_messages(db_session)
            for i in range(start, end):
                message_id = all_messages[i].id
                stmt = delete(AgentConversationHistory).where(
                    AgentConversationHistory.id == message_id
                )
                await db_session.execute(stmt)
            await db_session.commit()

    async def clear(self) -> None:
        async with self.db_session_maker() as db_session:
            stmt = delete(AgentConversationHistory).where(
                AgentConversationHistory.session_id == self.session_id
            )
            await db_session.execute(stmt)
            await db_session.commit()

    async def _get_all_messages(
        self, db_session: "AsyncSession"
    ) -> List[AgentConversationHistory]:
        stmt = (
            select(AgentConversationHistory)
            .where(
                AgentConversationHistory.session_id == self.session_id,
            )
            .order_by(AgentConversationHistory.id)
        )
        result = await db_session.execute(stmt)
        return list(result.scalars().all())

    async def _message_id_by_index(self, db_session: "AsyncSession", index: int) -> int:
        all_messages = await self._get_all_messages(db_session)
        return int(all_messages[index].id)
