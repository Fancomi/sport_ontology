import argostranslate.package
import argostranslate.translate

# 安装语言包（首次使用）
argostranslate.package.update_package_index()
available_packages = argostranslate.package.get_available_packages()
package_to_install = next(
    filter(lambda x: x.from_code == "en" and x.to_code == "zh", available_packages)
)
argostranslate.package.install_from_path(package_to_install.download())

# 执行翻译
translated_text = argostranslate.translate.translate("Hello World", "en", "zh")
print(translated_text)  # 输出：你好世界