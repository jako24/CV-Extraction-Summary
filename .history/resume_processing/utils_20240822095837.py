import re
import os
import json
import openai
import pytesseract
from pdf2image import convert_from_path
import cv2
import tempfile
import json
from dateutil import parser
from dateutil.relativedelta import relativedelta
import easyocr
import requests
from langchain_community.document_loaders import PyPDFLoader
import streamlit as st
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import numpy



def clean_text(extracted_text):
    # Remove dots and extra whitespace
    cleaned_text = re.sub(r'\s*\.\s*', ' ', extracted_text)
    return cleaned_text
