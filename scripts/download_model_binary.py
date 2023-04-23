#!/usr/bin/env python
import os
import sys
import time
import yaml
import urllib
import hashlib
import argparse

required_keys = ['caffemodel', 'caffemodel_