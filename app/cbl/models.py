from app.database import Base, int_pk
from sqlalchemy.orm import Mapped

class Request(Base):
    id: Mapped[int_pk]
    pick_up_x: Mapped[float]
    pick_up_y: Mapped[float]
    drop_off_x: Mapped[float]
    drop_off_y: Mapped[float]
    priority: Mapped[int]
    taken: Mapped[bool]

    def __str__(self):
        return (f"{self.__class__.__name__}(id={self.id}, pick_up={self.pick_up}, "
                f"drop_off={self.drop_off}, priority={self.priority})")