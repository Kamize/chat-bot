import os
import streamlit as st
from app import show_bot
from dotenv import load_dotenv

load_dotenv()

def main():
    show_bot()
    
if __name__ == "__main__":
    main()