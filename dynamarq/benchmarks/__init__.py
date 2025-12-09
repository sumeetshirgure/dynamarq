from .states import (
        GHZ,
        GHZReset,
    )

from .gates import (
        CNOTLadder,
        Fanout,
        LongRangeCNOT,
        LongRangeCNOTSparse,
    )

from .algorithms import (
        TFIM,
        IPE,
        QFT,
        PartialQFT,
    )

from .eccs import (
        RepetitionCode,
        FiveQubitCode,
    )

__all__ = [
        "GHZ",
        "GHZReset",
         "CNOTLadder",
        "Fanout",
        "LongRangeCNOT",
        "LongRangeCNOTSparse",
        "TFIM",
        "IPE",
        "QFT",
        "PartialQFT",
        "RepetitionCode",
        "FiveQubitCode",
        ]


def get_testbench() :
    bench = []

    GHZ_params = [2, 3, 5, 10, 15, 20, 25, 30]
    for param in GHZ_params : bench.append(GHZ.GHZ(param))

    GHZReset_params = [3, 5, 11, 15, 21, 25, 29]
    for param in GHZReset_params : bench.append(GHZReset.GHZReset(param))

    dfe_params = [2, 3, 5, 10, 15, 20, 25, 30]
    for param in dfe_params : bench.append(CNOTLadder.CNOTLadder(param))
    for param in dfe_params : bench.append(Fanout.Fanout(param))
    for param in dfe_params : bench.append(LongRangeCNOT.LongRangeCNOT(param))
    for param in dfe_params : bench.append(LongRangeCNOTSparse.LongRangeCNOTSparse(param))

    bench.append(RepetitionCode.RepetitionCode(3))
    bench.append(RepetitionCode.RepetitionCode(5))
    bench.append(FiveQubitCode.FiveQubitCode())

    qft_params = [2, 3, 5, 10, 15, 20]
    for param in qft_params : bench.append(QFT.QFT(param))
    for param in qft_params : bench.append(PartialQFT.PartialQFT(param))

    bench.append(IPE.IPE(int('11', 2), 2))
    bench.append(IPE.IPE(int('101', 2), 3))
    bench.append(IPE.IPE(int('10101', 2), 5))
    bench.append(IPE.IPE(int('1010101010', 2), 10))

    tfim_params = [ (3, 2), (3, 5), (3, 20), (5, 2), (5, 5), (5, 20),
                   (10, 2), (10, 5), (10, 20), (30, 2), (30, 5), (30, 20) ]
    for sites, steps in tfim_params : bench.append(TFIM.TFIM(sites, steps))

    return bench
