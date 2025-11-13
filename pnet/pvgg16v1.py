from predify.modules import PCoderN
from predify.networks import PNetSeparateHP
from torch.nn import Sequential, ReLU, Linear


class pvgg16v1SeparateHP(PNetSeparateHP):
    def __init__(
        self,
        backbone,
        build_graph=False,
        random_init=False,
        ff_multiplier=(0.2, 0.4, 0.4),
        fb_multiplier=(0.05, 0.1, 0.1),
        er_multiplier=(0.01, 0.01, 0.01),
    ):
        super().__init__(
            backbone,
            3,
            build_graph,
            random_init,
            ff_multiplier,
            fb_multiplier,
            er_multiplier,
        )

        # PCoder number 1
        pmodule = Linear(in_features=4096, out_features=25088, bias=True)
        self.pcoder1 = PCoderN(pmodule, True, self.random_init)

        def fw_hook1(m, m_in, m_out):
            ##manualy change self.input_mem to m_in
            e = self.pcoder1(
                ff=m_out,
                fb=self.pcoder2.prd,
                target=m_in[0],
                build_graph=self.build_graph,
                ffm=self.ffm1,
                fbm=self.fbm1,
                erm=self.erm1,
            )
            return e[0]

        self.backbone.classifier[0].register_forward_hook(fw_hook1)

        # PCoder number 2
        pmodule = Linear(in_features=4096, out_features=4096, bias=True)
        self.pcoder2 = PCoderN(pmodule, True, self.random_init)

        def fw_hook2(m, m_in, m_out):
            e = self.pcoder2(
                ff=m_out,
                fb=self.pcoder3.prd,
                target=self.pcoder1.rep,
                build_graph=self.build_graph,
                ffm=self.ffm2,
                fbm=self.fbm2,
                erm=self.erm2,
            )
            return e[0]

        self.backbone.classifier[3].register_forward_hook(fw_hook2)

        # PCoder number 3
        pmodule = Linear(in_features=1000, out_features=4096, bias=True)
        self.pcoder3 = PCoderN(pmodule, False, self.random_init)

        def fw_hook3(m, m_in, m_out):
            e = self.pcoder3(
                ff=m_out,
                fb=None,
                target=self.pcoder2.rep,
                build_graph=self.build_graph,
                ffm=self.ffm3,
                fbm=self.fbm3,
                erm=self.erm3,
            )
            return e[0]

        self.backbone.classifier[6].register_forward_hook(fw_hook3)
