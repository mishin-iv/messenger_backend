from fastapi import APIRouter

from app.cbl.dao import RequestsDAO
from app.cbl.schemas import SRequest

cbl_router = APIRouter(prefix='/api', tags=['Requests database for CBL project'])


@cbl_router.post('/add')
async def add_request(request: SRequest) -> dict:
    await RequestsDAO.add_request(**request.model_dump())
    return {'message': 'request added successfully', 'data': request.model_dump()}


@cbl_router.get('/list')
async def list_requests():
    return await RequestsDAO.return_all()


@cbl_router.post('/remove')
async def remove_request(del_id: int) -> dict:
    check = await RequestsDAO.find_by_id(del_id)
    if check:
        await RequestsDAO.remove_request(del_id)
        return {'message': 'success', 'deleted_id': del_id}
    else:
        return {'message': 'request not found', 'deleted_id': None}
