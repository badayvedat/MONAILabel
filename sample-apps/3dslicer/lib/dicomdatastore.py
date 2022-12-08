# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import hashlib
import logging
import os
import pathlib
import shutil
from typing import Any, Dict, Iterator, List, Optional, Tuple
import time

import requests
from cachetools import TTLCache, cached
from dicomweb_client import DICOMwebClient
from pydicom.dataset import Dataset

from monailabel.config import settings
from monailabel.datastore.local import LocalDatastore, LocalDatastoreModel
from monailabel.interfaces.datastore import Datastore, DefaultLabelTag
from monailabel.datastore.utils.convert import dicom_to_nifti

from watchdog.events import PatternMatchingEventHandler
from watchdog.observers import Observer

logger = logging.getLogger(__name__)

class DICOMLocalDataStore(LocalDatastore):
    def __init__(
        self,
        datastore_path: str,
        images_dir: str = ".",
        labels_dir: str = "labels",
        datastore_config: str = "datastore_v2.json",
        extensions=(),
        auto_reload=False,
    ):
        self._datastore_path = datastore_path
        self._datastore_config_path = os.path.join(datastore_path, datastore_config)
        self._extensions = [extensions] if isinstance(extensions, str) else extensions
        self._ignore_event_count = 0
        self._ignore_event_config = False
        self._config_ts = 0
        self._auto_reload = auto_reload

        logging.getLogger("filelock").setLevel(logging.ERROR)

        logger.info(f"Auto Reload: {auto_reload}; Extensions: {self._extensions}")

        os.makedirs(self._datastore_path, exist_ok=True)

        self._lock_file = os.path.join(datastore_path, ".lock")
        self._datastore: LocalDatastoreModel = LocalDatastoreModel(
            name="new-dataset", description="New Dataset", images_dir=images_dir, labels_dir=labels_dir
        )
        self._datastore.base_path = self._datastore_path
        self._init_from_datastore_file(throw_exception=True)

        os.makedirs(self._datastore.image_path(), exist_ok=True)
        os.makedirs(self._datastore.label_path(None), exist_ok=True)
        os.makedirs(self._datastore.label_path(DefaultLabelTag.FINAL), exist_ok=True)
        os.makedirs(self._datastore.label_path(DefaultLabelTag.ORIGINAL), exist_ok=True)

        # reconcile the loaded datastore file with any existing files in the path
        self._reconcile_datastore()

        if auto_reload:
            logger.info("Start observing external modifications on datastore (AUTO RELOAD)")
            # Image Dir
            include_patterns = [f"{self._datastore.image_path()}{os.path.sep}[a-z][a-z]*[0-9]"]

            # Label Dir(s)
            label_dirs = self._datastore.labels_path()
            label_dirs[DefaultLabelTag.FINAL] = self._datastore.label_path(DefaultLabelTag.FINAL)
            label_dirs[DefaultLabelTag.ORIGINAL] = self._datastore.label_path(DefaultLabelTag.ORIGINAL)
            for label_dir in label_dirs.values():
                include_patterns.extend(f"{label_dir}{os.path.sep}{ext}" for ext in [*extensions])

            # Config
            include_patterns.append(self._datastore_config_path)

            self._handler = PatternMatchingEventHandler(patterns=include_patterns)
            self._handler.on_created = self._on_any_event
            self._handler.on_deleted = self._on_any_event
            self._handler.on_modified = self._on_modify_event

            try:
                self._ignore_event_count = 0
                self._ignore_event_config = False
                self._observer = Observer()
                self._observer.schedule(self._handler, recursive=True, path=self._datastore_path)
                self._observer.start()
            except OSError as e:
                logger.error(
                    "Failed to start File watcher. "
                    "Local datastore will not update if images and labels are moved from datastore location."
                )
                logger.error(str(e))

    def get_image_uri(self, image_id: str) -> str:
        """
        Retrieve image uri based on image id

        :param image_id: the desired image's id
        :return: return the image uri
        """
        obj = self._datastore.objects.get(image_id)
        name = image_id if obj else ""
        output_file_nifti = dicom_to_nifti(os.path.realpath(os.path.join(self._datastore.image_path(),name))) if obj else ""

        logger.info(f"converted dicom {output_file_nifti}")

        return str(os.path.realpath(output_file_nifti)) if obj else ""