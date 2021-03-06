# Generated by the protocol buffer compiler.  DO NOT EDIT!
# sources: tensorbeat/sarosh_gen.proto
# plugin: python-betterproto
from dataclasses import dataclass
from typing import Dict, List, Optional

import betterproto
from betterproto.grpc.grpclib_server import ServiceBase
import grpclib


@dataclass(eq=False, repr=False)
class GenerateMusicRequest(betterproto.Message):
    notes: List[str] = betterproto.string_field(1)

    def __post_init__(self) -> None:
        super().__post_init__()


@dataclass(eq=False, repr=False)
class GenerateMusicResponse(betterproto.Message):
    notes: List[str] = betterproto.string_field(1)

    def __post_init__(self) -> None:
        super().__post_init__()


class SaroshGeneratorStub(betterproto.ServiceStub):
    async def generate_music(
        self, *, notes: Optional[List[str]] = None
    ) -> "GenerateMusicResponse":
        notes = notes or []

        request = GenerateMusicRequest()
        request.notes = notes

        return await self._unary_unary(
            "/tensorbeat.sarosh_gen.SaroshGenerator/GenerateMusic",
            request,
            GenerateMusicResponse,
        )


class SaroshGeneratorBase(ServiceBase):
    async def generate_music(
        self, notes: Optional[List[str]]
    ) -> "GenerateMusicResponse":
        raise grpclib.GRPCError(grpclib.const.Status.UNIMPLEMENTED)

    async def __rpc_generate_music(self, stream: grpclib.server.Stream) -> None:
        request = await stream.recv_message()

        request_kwargs = {
            "notes": request.notes,
        }

        response = await self.generate_music(**request_kwargs)
        await stream.send_message(response)

    def __mapping__(self) -> Dict[str, grpclib.const.Handler]:
        return {
            "/tensorbeat.sarosh_gen.SaroshGenerator/GenerateMusic": grpclib.const.Handler(
                self.__rpc_generate_music,
                grpclib.const.Cardinality.UNARY_UNARY,
                GenerateMusicRequest,
                GenerateMusicResponse,
            ),
        }
