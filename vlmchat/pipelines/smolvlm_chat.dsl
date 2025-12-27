{
        [camera(type="none"), input() -> break_on(code=1)] 
        -> history(id="hist", format="simple") 
        -> debug()
        -> smolvlm(system_prompt="You are a helpful vision assistant.") 
        -> history(id="hist") 
        -> debug()
        -> output()
        -> cleanup(remove_types="text")
}
