from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.by import By
import langchain
from langchain.text_splitter import RecursiveCharacterTextSplitter
import time
import socket
import threading
import os
import pyperclip as pc

def split_text_into_chunks(text, max_char):
    # i think the actual model takes more but google puts a limit on the input in the browser
    # 9910 is the max i found that works
    """Split text into chunks below a maximum number of tokens.
    Actual GPT-3.5-turbo and PaLM2 models can take slightly more than this limit.
    Though we will stick to this for consistency.

    Args:
        text: The text to split.
        max_tokens: The maximum number of tokens per chunk.

    Returns:
        A list of text chunks, each with less than or equal to max_tokens.
    """

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_char,
        chunk_overlap=20, # overlap is usually done by the MapReduce method to improve performance
    )
    chunks = text_splitter.split_text(text)
    return chunks

class BardAutomation:

    """
    Adapted from https://github.com/Michelangelo27/chatgpt_selenium_automation/blob/master/handler/chatgpt_selenium_automation.py:
    1. Updated for support of latest selenium version
    2. Removed chrome_driver_path parameter
    3. Added getNextPage method
    4. Adapted it to Bard PaLM2
    """

    def __init__(self, chrome_path):
        """
        This constructor automates the following steps:
        1. Open a Chrome browser with remote debugging enabled at a specified URL.
        2. Prompt the user to complete the log-in/registration/human verification, if required.
        3. Connect a Selenium WebDriver to the browser instance after human verification is completed.

        :param chrome_path: file path to chrome.exe (ex. C:\\Users\\User\\...\\chromedriver.exe)
        :param chrome_driver_path: file path to chrome.exe (ex. C:\\Users\\User\\...\\chromedriver.exe)
        """

        self.chrome_path = chrome_path

        url = r"https://bard.google.com/chat"
        free_port = self.find_available_port()
        self.launch_chrome_with_remote_debugging(free_port, url)
        self.wait_for_human_verification()
        self.driver = self.setup_webdriver(free_port)



    def find_available_port(self):
        """ This function finds and returns an available port number on the local machine by creating a temporary
            socket, binding it to an ephemeral port, and then closing the socket. """

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            return s.getsockname()[1]



    def launch_chrome_with_remote_debugging(self, port, url):
        """ Launches a new Chrome instance with remote debugging enabled on the specified port and navigates to the
            provided url """

        def open_chrome():
            chrome_cmd = f"{self.chrome_path} --remote-debugging-port={port} --user-data-dir=remote-profile {url}"
            os.system(chrome_cmd)

        chrome_thread = threading.Thread(target=open_chrome)
        chrome_thread.start()

    def setup_webdriver(self, port):
        """  Initializes a Selenium WebDriver instance, connected to an existing Chrome browser
             with remote debugging enabled on the specified port"""

        chrome_options = webdriver.ChromeOptions()
        chrome_options.add_experimental_option("debuggerAddress", f"127.0.0.1:{port}")

        # driver = webdriver.Chrome(executable_path=self.chrome_driver_path, options=chrome_options)
        # updated for support of latest selenium version
        driver = webdriver.Chrome(service=ChromeService(), options=chrome_options)
        return driver


    def send_prompt_to_bard(self, prompt, add_prompt="TLDR in 200 words, without bullet points:"):
        """ Sends a message to Bard and waits for 20 seconds for the response """

        input_box = self.driver.find_element(by=By.XPATH, value="//div[@data-placeholder='Enter a prompt here']")

        # print the html parent of the input box
        escaped_prompt = prompt.replace("\"", "")
        escaped_prompt = escaped_prompt.replace("\'", "\\'")
        
        escaped_prompt = add_prompt + escaped_prompt
        escaped_prompt = escaped_prompt.replace("\n", " ")
        escaped_prompt = escaped_prompt.strip()

        print(escaped_prompt)

        pc.copy(escaped_prompt)
        input_box.send_keys(Keys.CONTROL, 'v') # text too long to send_keys, so copy instead

        # wait for copy to finish
        time.sleep(2)
        
        print("Sending prompt to Bard...")
        input_box.send_keys(Keys.ENTER)
        print("ENTER")

        # if enter doesnt do anhthing means bard is still returning smth
        # wait 10s before trying to enter again
        # check if the input bow has any text

        while True:
            if input_box.text == "":
                break
            else:
                time.sleep(5)

                # retries eneter every 5s
                input_box.send_keys(Keys.ENTER)
                print("ENTER")

        # bard is really quite slow with longer text
        # waiting too short means we won't be able to get the output response
        time.sleep(40)


    def return_last_response(self):
        """ :return: the text of the last bard response """
        response_elements = self.driver.find_elements(by=By.CSS_SELECTOR, value='.response-content')
        print(response_elements)
        return response_elements[-1].text



    def wait_for_human_verification(self):
        print("You need to manually complete the log-in or the human verification if required.")

        while True:
            user_input = input(
                "Enter 'y' if you have completed the log-in or the human verification, or 'n' to check again: ").lower()

            if user_input == 'y':
                print("Continuing with the automation process...")
                break
            elif user_input == 'n':
                print("Waiting for you to complete the human verification...")
                time.sleep(5)  # You can adjust the waiting time as needed
            else:
                print("Invalid input. Please enter 'y' or 'n'.")

    def getNextPage(self):
        newChatBtn = self.driver.find_element(by=By.CSS_SELECTOR, value="[data-test-id='new-chat']")
        newChatBtn.click()
        time.sleep(5)

    def quit(self):
        """ Closes the browser and terminates the WebDriver session."""
        print("Closing the browser...")
        self.driver.close()
        self.driver.quit()