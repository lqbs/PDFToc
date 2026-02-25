# -*- coding: utf-8 -*-

"""
The main GUI model of project.

"""

import os
import inspect
import sys
import traceback
import webbrowser

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMessageBox

from src.config import CONFIG
from src.convert import clean_clipboard_control_chars, convert_dir_text
from src.gui.base import TreeWidget
from src.gui.main_ui import Ui_PDFdir
from src.updater import is_updated
from src.pdf.bookmark import add_bookmark, check_bookmarks, get_bookmarks

# import qdarkstyle


QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)


def dynamic_base_class(instance, cls_name, new_class, **kwargs):
    instance.__class__ = type(cls_name, (new_class, instance.__class__), kwargs)
    return instance


class ControlButtonMixin(object):
    def set_control_button(self, min_button, exit_button):
        min_button.clicked.connect(self.showMinimized)
        exit_button.clicked.connect(self.close)


class AiExtractWorker(QtCore.QObject):
    succeeded = QtCore.pyqtSignal(str)
    failed = QtCore.pyqtSignal(str)
    done = QtCore.pyqtSignal()

    def __init__(self, pdf_path, start_page=1, end_page=20):
        super().__init__()
        self.pdf_path = pdf_path
        self.start_page = start_page
        self.end_page = end_page

    @QtCore.pyqtSlot()
    def run(self):
        try:
            try:
                from src.ai_toc import extract_toc_text
            except ImportError:
                raise RuntimeError("AI目录功能不可用，请确认 src/ai_toc.py 已正确接入。")
            except Exception as e:
                raise RuntimeError("AI目录模块初始化失败：%s" % (str(e).strip() or "未知错误"))

            extract_params = inspect.signature(extract_toc_text).parameters
            if "start_page" in extract_params and "end_page" in extract_params:
                toc_text = extract_toc_text(
                    self.pdf_path, start_page=self.start_page, end_page=self.end_page
                )
            else:
                toc_text = extract_toc_text(self.pdf_path)
            if not isinstance(toc_text, str) or not toc_text.strip():
                raise ValueError("模型响应为空，请检查 API 配置或重试。")
            self.succeeded.emit(toc_text.strip())
        except Exception as e:
            self.failed.emit(str(e).strip() or "未知错误")
        finally:
            self.done.emit()


class Main(QtWidgets.QMainWindow, Ui_PDFdir, ControlButtonMixin):
    def __init__(self, app, trans):
        super(Main, self).__init__()
        # self.setWindowFlags(Qt.FramelessWindowHint)
        # self.menuBar.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.app = app
        self.trans = trans
        self.setupUi(self)
        self.version = CONFIG.VERSION
        self.default_folder = CONFIG.DEFAULT_FOLDER
        self.setWindowTitle(
            "{name} {version}".format(name=CONFIG.APP_NAME, version=CONFIG.VERSION)
        )
        self.setWindowIcon(QtGui.QIcon("{icon}".format(icon=CONFIG.WINDOW_ICON)))
        self.dir_tree_widget = dynamic_base_class(
            self.dir_tree_widget, "TreeWidget", TreeWidget
        )
        self.dir_tree_widget.init_connect(parents=[self, self.dir_tree_widget])
        self.dir_tree_widget.fix_column()
        self._set_connect()
        self._set_action()
        self._set_unwritable()
        self._worker = None
        self._worker_thread = None

    def _set_connect(self):
        self.open_button.clicked.connect(self.open_file_dialog)
        self.export_button.clicked.connect(self.write_tree_to_pdf)
        if hasattr(self, "ai_extract_button"):
            self.ai_extract_button.clicked.connect(self.extract_dir_text_by_ai)
        self.level0_box.clicked.connect(self._change_level0_writable)
        self.level1_box.clicked.connect(self._change_level1_writable)
        self.level2_box.clicked.connect(self._change_level2_writable)
        self.level3_box.clicked.connect(self._change_level3_writable)
        self.level4_box.clicked.connect(self._change_level4_writable)
        self.level5_box.clicked.connect(self._change_level5_writable)
        for act in (
            self.dir_text_edit.textChanged,
            self.offset_edit.textChanged,
            self.level0_box.stateChanged,
            self.level1_box.stateChanged,
            self.level2_box.stateChanged,
            self.level3_box.stateChanged,
            self.level4_box.stateChanged,
            self.level5_box.stateChanged,
            self.level0_edit.textChanged,
            self.level1_edit.textChanged,
            self.level2_edit.textChanged,
            self.level3_edit.textChanged,
            self.level4_edit.textChanged,
            self.level5_edit.textChanged,
            self.unknown_level_box.currentIndexChanged,
            self.space_level_box.stateChanged,
            self.fix_non_seq_action.changed,
        ):
            act.connect(self.make_dir_tree)

    def _set_action(self):
        self.home_page_action.triggered.connect(self._open_home_page)
        self.help_action.triggered.connect(self._open_help_page)
        self.update_action.triggered.connect(self._open_update_page)
        self.english_action.triggered.connect(self.to_english)
        self.chinese_action.triggered.connect(self.to_chinese)

    def _set_unwritable(self):
        self.level0_edit.setEnabled(False)
        self.level1_edit.setEnabled(False)
        self.level2_edit.setEnabled(False)
        self.level3_edit.setEnabled(False)
        self.level4_edit.setEnabled(False)
        self.level5_edit.setEnabled(False)

    def _change_level0_writable(self):
        self.level0_edit.setEnabled(True if self.level0_box.isChecked() else False)

    def _change_level1_writable(self):
        self.level1_edit.setEnabled(True if self.level1_box.isChecked() else False)

    def _change_level2_writable(self):
        self.level2_edit.setEnabled(True if self.level2_box.isChecked() else False)

    def _change_level3_writable(self):
        self.level3_edit.setEnabled(True if self.level3_box.isChecked() else False)

    def _change_level4_writable(self):
        self.level4_edit.setEnabled(True if self.level4_box.isChecked() else False)

    def _change_level5_writable(self):
        self.level5_edit.setEnabled(True if self.level5_box.isChecked() else False)

    @staticmethod
    def _open_home_page():
        webbrowser.open(CONFIG.HOME_PAGE_URL, new=1)

    @staticmethod
    def _open_help_page():
        webbrowser.open(CONFIG.HELP_PAGE_URL, new=1)

    def _open_update_page(self):
        url = CONFIG.RELEASE_PAGE_URL
        try:
            updated = is_updated(url, self.version)
        except Exception:
            self.alert_msg("Check update failed", level="warn")
        else:
            if updated:
                self.show_status("Find new version", 3000)
                webbrowser.open(url, new=1)
            else:
                self.show_status("No update", 3000)
                self.alert_msg("No update")

    def show_status(self, msg, timeout=10 * 3600 * 1000):
        """Show message in status bar"""
        return self.statusbar.showMessage(msg, msecs=timeout)

    @staticmethod
    def alert_msg(msg, level="info", ok_action=None):
        box = QMessageBox()
        if level == "info":
            box.setIcon(QMessageBox.Information)
            box.setWindowTitle("Infomation")
        else:
            box.setIcon(QMessageBox.Warning)
            box.setWindowTitle("Warning")
        if ok_action:
            box.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
            box.buttonClicked.connect(ok_action)
        box.setText(msg)
        box.exec_()

    def to_english(self):
        self.trans.load("./language/en")
        self.app.installTranslator(self.trans)
        self.retranslateUi(self)

    def to_chinese(self):
        self.app.removeTranslator(self.trans)
        self.retranslateUi(self)

    @property
    def pdf_path(self):
        return self.pdf_path_edit.text()

    @property
    def dir_text(self):
        return self.dir_text_edit.toPlainText()

    @property
    def offset_num(self):
        offset = self.offset_edit.text()
        if isinstance(offset, str) and offset.lstrip("-").isdigit():
            return int(offset)
        return 0

    @property
    def level0_text(self):
        return self.level0_edit.text() if self.level0_box.isChecked() else None

    @property
    def level1_text(self):
        return self.level1_edit.text() if self.level1_box.isChecked() else None

    @property
    def level2_text(self):
        return self.level2_edit.text() if self.level2_box.isChecked() else None

    @property
    def level3_text(self):
        return self.level3_edit.text() if self.level3_box.isChecked() else None

    @property
    def level4_text(self):
        return self.level4_edit.text() if self.level4_box.isChecked() else None

    @property
    def level5_text(self):
        return self.level5_edit.text() if self.level5_box.isChecked() else None

    @property
    def other_level_index(self):
        return self.unknown_level_box.currentIndex()

    @property
    def level_by_space(self):
        return self.space_level_box.isChecked()

    @property
    def fix_non_seq(self):
        return self.fix_non_seq_action.isChecked()

    @property
    def keep_exist_dir(self):
        return self.keep_exist_dir_action.isChecked()

    @property
    def read_exist_dir(self):
        return self.read_exist_dir_action.isChecked()

    def open_file_dialog(self):
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "select PDF", directory=self.default_folder, filter="PDF (*.pdf)"
        )
        self.default_folder = os.path.dirname(filename)
        self.pdf_path_edit.setText(filename)

        exist_bookmarks = self.read_pdf_dir_text(filename)
        if exist_bookmarks and self.read_exist_dir:
            exist_bookmarks = clean_clipboard_control_chars(exist_bookmarks)
            self.dir_text_edit.setText(exist_bookmarks)
            self.space_level_box.setChecked(True)

    def tree_to_dict(self):
        return self.dir_tree_widget.to_dict()

    def make_dir_tree(self):
        self.dir_tree_widget.clear()
        index_dict = convert_dir_text(
            self.dir_text,
            self.offset_num,
            self.level0_text,
            self.level1_text,
            self.level2_text,
            self.level3_text,
            self.level4_text,
            self.level5_text,
            other=self.other_level_index,
            level_by_space=self.level_by_space,
            fix_non_seq=self.fix_non_seq,
        )
        top_idx = 0
        inserted_items = {}
        children = {}
        for i, con in index_dict.items():
            if "parent" in con:
                children[i] = con
            else:
                # Insert all top items
                tree_item = QtWidgets.QTreeWidgetItem(
                    [
                        con.get("title"),
                        str(con.get("num", 1)),
                        str(con.get("real_num", 1)),
                    ]
                )
                self.dir_tree_widget.insertTopLevelItem(top_idx, tree_item)
                inserted_items[i] = tree_item
                top_idx += 1
        # Insert all children items
        last_children_count = len(children) + 1
        while children and len(children) < last_children_count:
            keys = set(children.keys())
            for k in keys:
                con = children[k]
                p_idx = con["parent"]
                if p_idx in inserted_items:
                    p_item = inserted_items[p_idx]
                    tree_item = QtWidgets.QTreeWidgetItem(
                        [
                            con.get("title"),
                            str(con.get("num", 1)),
                            str(con.get("real_num", 1)),
                        ]
                    )
                    p_item.addChild(tree_item)
                    children.pop(k)
                    inserted_items[k] = tree_item
        for item in inserted_items.values():
            item.setExpanded(1)

    def pre_check(self, path, index_dict):
        try:
            check_bookmarks(path, index_dict, self.keep_exist_dir)
        except ValueError as e:
            self.alert_msg(str(e), level="Warning")

    def write_tree_to_pdf(self):
        try:
            index_dict = self.tree_to_dict()
            self.pre_check(self.pdf_path, index_dict)
            new_path = self.dict_to_pdf(self.pdf_path, index_dict, self.keep_exist_dir)
            self.alert_msg("%s Finished！" % new_path)
        except PermissionError:
            self.alert_msg("Permission denied！", level="warn")

    def extract_dir_text_by_ai(self):
        pdf_path = (self.pdf_path or "").strip()
        if not pdf_path:
            self.alert_msg("请先选择PDF文件。", level="warn")
            return
        if not os.path.isfile(pdf_path):
            self.alert_msg("PDF文件不存在，请重新选择。", level="warn")
            return
        start_page_raw = self._get_ai_start_page_raw()
        end_page_raw = self._get_ai_end_page_raw()
        if not start_page_raw or not end_page_raw:
            self.alert_msg("目录页开始/结束页不能为空。", level="warn")
            return
        if not start_page_raw.isdigit() or not end_page_raw.isdigit():
            self.alert_msg("目录页开始/结束页必须是正整数。", level="warn")
            return
        start_page = int(start_page_raw)
        end_page = int(end_page_raw)
        if start_page <= 0 or end_page <= 0:
            self.alert_msg("目录页开始/结束页必须大于0。", level="warn")
            return
        if start_page > end_page:
            self.alert_msg("目录页开始页不能大于结束页。", level="warn")
            return
        if self._worker_thread is not None and self._worker_thread.isRunning():
            self.show_status("正在处理，请稍候...", 3000)
            return

        self.show_status("正在自动获取目录文本，请稍候...", 5000)
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        if hasattr(self, "ai_extract_button"):
            self.ai_extract_button.setEnabled(False)
        self._worker_thread = QtCore.QThread(self)
        self._worker = AiExtractWorker(
            pdf_path, start_page=start_page, end_page=end_page
        )
        self._worker.moveToThread(self._worker_thread)

        self._worker_thread.started.connect(self._worker.run)
        self._worker.succeeded.connect(self._on_ai_extract_succeeded)
        self._worker.failed.connect(self._on_ai_extract_failed)
        self._worker.done.connect(self._on_ai_extract_done)
        self._worker.done.connect(self._worker_thread.quit)
        self._worker_thread.finished.connect(self._worker.deleteLater)
        self._worker_thread.finished.connect(self._worker_thread.deleteLater)
        self._worker_thread.finished.connect(self._clear_ai_extract_worker)
        self._worker_thread.start()

    def _on_ai_extract_succeeded(self, toc_text):
        self.dir_text_edit.setPlainText(toc_text)
        self.show_status("目录文本已自动填充。", 5000)

    def _on_ai_extract_failed(self, err_msg):
        self.show_status("自动获取目录失败。", 5000)
        self.alert_msg("自动获取目录失败：%s" % err_msg, level="warn")

    def _on_ai_extract_done(self):
        if QtWidgets.QApplication.overrideCursor() is not None:
            QtWidgets.QApplication.restoreOverrideCursor()

    def _clear_ai_extract_worker(self):
        if hasattr(self, "ai_extract_button"):
            self.ai_extract_button.setEnabled(True)
        self._worker = None
        self._worker_thread = None

    def _get_ai_start_page_raw(self):
        for attr_name in ("ai_page_start_edit", "ai_start_page_edit"):
            if hasattr(self, attr_name):
                return getattr(self, attr_name).text().strip()
        return "1"

    def _get_ai_end_page_raw(self):
        for attr_name in ("ai_page_end_edit", "ai_end_page_edit"):
            if hasattr(self, attr_name):
                return getattr(self, attr_name).text().strip()
        return "20"

    def closeEvent(self, event):
        if self._worker_thread is not None and self._worker_thread.isRunning():
            self.alert_msg("正在自动获取目录文本，请稍候完成后再关闭。", level="warn")
            event.ignore()
            return
        super(Main, self).closeEvent(event)

    @staticmethod
    def dict_to_pdf(pdf_path, index_dict, keep_exist_dir=False):
        return add_bookmark(pdf_path, index_dict, keep_exist_dir)

    @staticmethod
    def read_pdf_dir_text(pdf_path):
        return "\n".join(get_bookmarks(pdf_path))


def run():
    app = QtWidgets.QApplication(sys.argv)
    # app.setStyle('fusion')
    # app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    trans = QtCore.QTranslator()
    # trans.load("./gui/en")
    # app.installTranslator(trans)
    window = Main(app, trans)
    window.show()
    sys.exit(app.exec_())


sys._excepthook = sys.excepthook


def exception_hook(exctype, value, exc_traceback):
    sys._excepthook(exctype, value, exc_traceback)
    error_message = "".join(traceback.format_exception(exctype, value, exc_traceback))
    QMessageBox.critical(None, "Unhandled Exception", error_message)
    # Optionally, call the original excepthook
    if hasattr(sys, "_excepthook"):
        sys._excepthook(exctype, value, exc_traceback)


sys.excepthook = exception_hook


if __name__ == "__main__":
    run()
