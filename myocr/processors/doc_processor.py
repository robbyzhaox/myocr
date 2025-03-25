# from ..base import BaseProcessor


# class DocProcessorChainGenerator(BaseProcessor):

#     def __init__(self, processors):
#         pass


# class DocPreProcessors(DocProcessorChainGenerator):
#     def __init__(self, processors):
#         self.prcessors = processors

#     def __call__(self, input, **kwargs):

#         for processor in self.prcessors:
#             processor.process(input, **kwargs)


# class DocPostProcessors(DocProcessorChainGenerator):
#     def __init__(self, processors):
#         self.prcessors = processors

#     def __call__(self, input, **kwargs):
#         pass
