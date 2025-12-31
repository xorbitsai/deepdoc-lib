#
#  Copyright 2025 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

"""
Simple Chinese surname detection utility for resume parsing.
"""

# Common Chinese surnames (most frequent ones)
CHINESE_SURNAMES = {
    # Single character surnames (most common)
    '王', '李', '张', '刘', '陈', '杨', '赵', '黄', '周', '吴',
    '徐', '孙', '胡', '朱', '高', '林', '何', '郭', '马', '罗',
    '梁', '宋', '郑', '谢', '韩', '唐', '冯', '于', '董', '萧',
    '程', '曹', '袁', '邓', '许', '傅', '沈', '曾', '彭', '吕',
    '苏', '卢', '蒋', '蔡', '贾', '丁', '魏', '薛', '叶', '阎',
    '余', '潘', '杜', '戴', '夏', '钟', '汪', '田', '任', '姜',
    '范', '方', '石', '姚', '谭', '廖', '邹', '熊', '金', '陆',
    '郝', '孔', '白', '崔', '康', '毛', '邱', '秦', '江', '史',
    '顾', '侯', '邵', '孟', '龙', '万', '段', '漕', '钱', '汤',
    '尹', '黎', '易', '常', '武', '乔', '贺', '赖', '龚', '文',

    # Double character surnames (common compound surnames)
    '欧阳', '太史', '端木', '上官', '司马', '东方', '公孙', '万俟', '闻人',
    '夏侯', '诸葛', '尉迟', '公西', '澹台', '赫连', '皇甫', '宗政', '濮阳',
    '公冶', '太叔', '申屠', '公孙', '慕容', '仲孙', '钟离', '长孙', '司徒',
    '鲜于', '司空', '宇文', '长孙', '慕容', '司徒'
}


class SurnameChecker:
    """Chinese surname checker utility"""

    def __init__(self):
        self.surnames = CHINESE_SURNAMES

    def isit(self, text: str) -> bool:
        """
        Check if the given text starts with a Chinese surname.

        Args:
            text: Text to check

        Returns:
            bool: True if text starts with a Chinese surname
        """
        if not text or not isinstance(text, str):
            return False

        text = text.strip()

        # Check single character surnames
        if len(text) >= 1 and text[0] in self.surnames:
            return True

        # Check double character surnames
        if len(text) >= 2:
            double_surname = text[:2]
            if double_surname in self.surnames:
                return True

        return False


# Global instance for backward compatibility
surname = SurnameChecker()


def is_chinese_surname(text: str) -> bool:
    """
    Check if text starts with a Chinese surname.

    Args:
        text: Text to check

    Returns:
        bool: True if starts with Chinese surname
    """
    return surname.isit(text)
