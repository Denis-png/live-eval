"""Shared ERRANT annotator — loaded once and reused across metric modules."""
import errant

annotator = errant.load("en")
