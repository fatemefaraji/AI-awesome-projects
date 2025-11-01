# Enhanced Music Composition System with Advanced Features

!pip install langchain langchain_core langgraph langchain_community langchain_groq music21 pretty_midi pygame
!apt install fluidsynth -qq

import os
from typing import TypedDict, Dict, List, Optional, Any
from langgraph.graph import StateGraph, END
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.runnables.graph import MermaidDrawMethod
from IPython.display import display, Image, Audio, Markdown
import music21
import tempfile
import random
import json
from datetime import datetime
import base64
import pretty_midi
import pygame
import io

# Enhanced State Definition with Comprehensive Musical Parameters
class MusicState(TypedDict):
    musician_input: str
    key: str
    tempo: int
    time_signature: str
    instruments: List[str]
    style: str
    emotion: str
    complexity: str
    duration_bars: int
    structure: Dict[str, Any]
    melody: str
    harmony: str
    rhythm: str
    bass_line: str
    counterpoint: str
    dynamics: str
    composition: str
    midi_file: str
    music_xml: str
    audio_file: str
    metadata: Dict[str, Any]
    analysis: Dict[str, Any]

# Configuration Management
class MusicConfig:
    def __init__(self):
        self.available_keys = [
            'C major', 'C minor', 'G major', 'G minor', 'D major', 'D minor',
            'A major', 'A minor', 'E major', 'E minor', 'F major', 'F minor',
            'Bb major', 'Bb minor', 'Eb major', 'Eb minor'
        ]
        self.available_tempos = {
            'largo': 50, 'adagio': 70, 'andante': 90, 'moderato': 110,
            'allegro': 130, 'presto': 160, 'prestissimo': 180
        }
        self.available_instruments = {
            'piano': music21.instrument.Piano,
            'violin': music21.instrument.Violin,
            'viola': music21.instrument.Viola,
            'cello': music21.instrument.Cello,
            'flute': music21.instrument.Flute,
            'clarinet': music21.instrument.Clarinet,
            'guitar': music21.instrument.Guitar,
            'string_quartet': [music21.instrument.Violin, music21.instrument.Violin, 
                             music21.instrument.Viola, music21.instrument.Cello]
        }
        self.available_styles = [
            'baroque', 'classical', 'romantic', 'impressionist', 
            'contemporary', 'jazz', 'minimalist', 'film_score'
        ]
        self.available_emotions = [
            'joyful', 'sorrowful', 'mysterious', 'energetic', 'calm', 
            'dramatic', 'heroic', 'nostalgic', 'tension', 'resolution'
        ]
        self.complexity_levels = ['simple', 'moderate', 'complex', 'virtuoso']

