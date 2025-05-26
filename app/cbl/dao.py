from sqlalchemy import select, delete
from sqlalchemy.exc import SQLAlchemyError

from app.database import async_session_maker
from app.cbl.models import Request

class RequestsDAO:
    model = Request

    @classmethod
    async def add_request(cls, **values):
        async with async_session_maker() as session:
            async with session.begin():
                new_instance = cls.model(**values)
                session.add(new_instance)
                try:
                    await session.commit()
                except SQLAlchemyError as e:
                    await session.rollback()
                    raise e
                return new_instance.id

    @classmethod
    async def return_all(cls):
        async with async_session_maker() as session:
            query = select(cls.model).order_by(cls.model.priority.desc())
            result = await session.execute(query)
            print(result.scalars())
            return result.scalars().all()

    @classmethod
    async def remove_request(cls, remove_id: int):
        async with async_session_maker() as session:
            async with session.begin():
                query = delete(cls.model).where(cls.model.id == remove_id)
                await session.execute(query)
                try:
                    await session.commit()
                except SQLAlchemyError as e:
                    await session.rollback()
                    raise e
                return

    @classmethod
    async def find_by_id(cls, search_id: int):
        async with async_session_maker() as session:
            query = select(cls.model).filter_by(id=search_id)
            result = await session.execute(query)
            return result.scalar_one_or_none()