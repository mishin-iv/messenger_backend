import numpy as np
from fastapi import APIRouter

from app.cbl.dao import RequestsDAO
from app.cbl.schemas import SRequest

cbl_router = APIRouter(prefix='/api', tags=['Requests database for CBL project'])


@cbl_router.post('/add')
async def add_request(request: SRequest) -> dict:
    request_id = await RequestsDAO.add_request(**request.model_dump())
    return {'message': 'request added successfully', 'id': request_id, 'data': request.model_dump()}


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

@cbl_router.post('/take')
async def take_request(req_id: int) -> dict:
    check = await RequestsDAO.find_by_id(req_id)
    if check:
        await RequestsDAO.set_taken(req_id=req_id)
        return {'message': 'success', 'taken_id': req_id}
    else:
        return {'message': 'request not found', 'taken_id': None}


@cbl_router.get('/danger')
async def danger_request(x: int, y: int) -> dict:
    danger_map = np.zeros((y * 5, x * 5), dtype=int)
    for i in range(y * 5):
        for j in range(x * 5):
            if (2 * x) <= j < (3 * x) and (2 * y) <= i < (3 * y):
                danger_map[i][j] = 2
            elif x <= j < (4 * x) and y <= i < (4 * y):
                danger_map[i][j] = 1
    return {'message': 'success', 'map': danger_map.tolist()}
