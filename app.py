import streamlit as st
from bot import graph_with_order_tools, WELCOME_MSG

# Streamlit UI
def show_bot():
    st.title("BaristaBot Cafe")
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": WELCOME_MSG}]
    if "graph_state" not in st.session_state:
        st.session_state.graph_state = {"messages": []}

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What can I get for you?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            # We are using the full history as the graph input for each turn
            # to maintain context correctly.
            user_message_tuple = ("user", prompt)
            st.session_state.graph_state['messages'].append(user_message_tuple)
            
            # Here we invoke the full graph with the entire conversation history
            # stored in the session state.
            response_state = graph_with_order_tools.invoke(st.session_state.graph_state)
            
            # The graph returns the updated state, we store it.
            st.session_state.graph_state = response_state
            
            # The Streamlit UI only displays the final response message.
            bot_response = response_state['messages'][-1].content
            
            st.markdown(bot_response)
            
            # Add the final bot response to the Streamlit messages history.
            st.session_state.messages.append(
                {"role": "assistant", "content": bot_response}
            )