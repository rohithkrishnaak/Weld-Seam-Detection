"""Fuzzy Number Processing Module for Weld Seam Detection.

Provides fuzzy-logic-based segmentation, morphological cleaning, and
sub-pixel center extraction for laser stripe profiles on reflective
metallic surfaces.

Modules
-------
fuzzy_pipeline
    Dual-path orchestrator (classical vs. IT2FLS-enhanced).
type2_barycentric
    Interval Type-2 Fuzzy Logic (IT2FLS) with Enhanced Karnik-Mendel.
fcm_segmentation
    Fuzzy C-Means clustering for laser stripe segmentation.
fuzzy_barycentric
    Fuzzy gray barycentric center extraction.
tfn_extraction
    Triangular Fuzzy Number (TFN) based extraction with uncertainty.
fuzzy_morphology
    Fuzzy morphological opening and closing.
"""

from fuzzy.fuzzy_pipeline import FuzzyPipeline, FuzzyResult
from fuzzy.type2_barycentric import IT2FLSExtractor
from fuzzy.fcm_segmentation import FuzzyCMeans
from fuzzy.fuzzy_barycentric import FuzzyGrayBarycentric
from fuzzy.tfn_extraction import TFNCenterExtractor
from fuzzy.fuzzy_morphology import FuzzyMorphology

__all__ = [
    "FuzzyPipeline",
    "FuzzyResult",
    "IT2FLSExtractor",
    "FuzzyCMeans",
    "FuzzyGrayBarycentric",
    "TFNCenterExtractor",
    "FuzzyMorphology",
]
