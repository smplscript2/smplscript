#############
# Imports   #
#############
from strings_with_arrows import string_with_arrows
import string

#############
# Constants #
#############

digits = '0123456789'
letters = string.ascii_letters
ld = letters + digits

########################
# Errors               #
########################

class Error:
    def __init__(self, ps, pe, en, dt):
        self.ps = ps
        self.pe = pe
        self.en = en
        self.dt = dt
    
    def as_string(self):
        res = f'{self.en}: {self.dt}'
        res += f'\nFile {self.ps.fn}, line {self.ps.ln+1}'
        res += '\n\n' + string_with_arrows(self.ps.ftxt, self.ps, self.pe)
        return res

class IllegalCharError(Error):
    def __init__(self, ps, pe, dt):
        super().__init__(ps, pe, 'Illegal Character', dt)

class ExpectedCharError(Error):
    def __init__(self, ps, pe, dt):
        super().__init__(ps, pe, 'Expected Character', dt)

class InvalidSyntaxError(Error):
    def __init__(self, ps, pe, dt=''):
        super().__init__(ps, pe, 'Invalid Syntax', dt)

class RTError(Error):
    def __init__(self, ps, pe, dt, context):
        super().__init__(ps, pe, 'Runtime error', dt)
        self.context = context
    
    def as_string(self):
        res = self.generate_traceback()
        res += f'{self.en}: {self.dt}'
        res += '\n\n' + string_with_arrows(self.ps.ftxt, self.ps, self.pe)
        return res
    
    def generate_traceback(self):
        res = ''
        pos = self.ps
        ctx = self.context

        while ctx:
            res = f'\tFile {pos.fn}, line {pos.ln+1}, in {ctx.dn}\n' + res
            pos = ctx.pep
            ctx = ctx.parent
        return 'Traceback (most recent call last):\n' + res

########################
# Position             #
########################

class Position:
    def __init__(self, idx, ln, col, fn, ftxt):
        self.idx = idx
        self.ln = ln
        self.col = col
        self.fn = fn
        self.ftxt = ftxt
    
    def advance(self, current_char=None):
        self.idx += 1
        self.col += 1

        if current_char == '\n':
            self.ln += 1
            self.col = 0
    
        return self
    
    def copy(self):
        return Position(self.idx, self.ln, self.col, self.fn, self.ftxt)

########################
# Tokens               #
########################

tt_int      = 'Int'
tt_float    = 'float'
tt_ident    = 'identifier'
tt_keyword  = "keyword"
tt_plus     = 'plus'
tt_minus    = 'minus'
tt_mult     = 'mult'
tt_div      = 'div'
tt_pow      = 'pow'
tt_eq       = 'equals'
tt_lparen   = 'lparen'
tt_rparen   = 'rparen'
tt_ee       = 'ee'
tt_ne       = 'ne'
tt_gt       = 'gt'
tt_lt       = 'lt'
tt_lte      = 'lte'
tt_gte      = 'gte'
tt_comma    = 'comma'
tt_arrow    = 'arrow'
tt_eof      = "EOF"
tt_comment  = 'comment'

keywords = [
    'variable',
    'and',
    'or',
    'not',
    'if',
    'elif',
    'else',
    'then',
    'while',
    'to',
    'for',
    'step',
    'func'
]

class Token:
    def __init__(self, type_, value=None, ps=None, pe=None):
        self.type = type_
        self.value = value 
        if ps:
            self.ps = ps.copy()
            self.pe = ps.copy()
            self.pe.advance()
        if pe:
            self.pe = pe.copy()
    
    def matches(self, type_, value):
        return self.type == type_ and self.value == value

    def __repr__(self):
        if self.value: return f'{self.type}: {self.value}'
        return f'{self.type}'

########################
# Lexer                #
########################

class Lexer:
    def __init__(self, fn, text):
        self.fn = fn
        self.text = text
        self.pos = Position(-1, 0, -1, self.fn, self.text)
        self.current_char = None
        self.advance()
    
    def advance(self):
        self.pos.advance(self.current_char)
        self.current_char = self.text[self.pos.idx] if self.pos.idx < len(self.text) else None
    
    def make_tokens(self):
        tokens = []

        while self.current_char != None:
            if self.current_char in ' \t':
                self.advance()
            elif self.current_char in digits:
                tokens.append(self.make_number())
            elif self.current_char in letters:
                tokens.append(self.make_ident())
            elif self.current_char == '+':
                tokens.append(Token(tt_plus, ps=self.pos))
                self.advance()
            elif self.current_char == '-':
                self.make_min_arrow()
            elif self.current_char == '*':
                tokens.append(Token(tt_mult, ps=self.pos))
                self.advance()
            elif self.current_char == '/':
                tokens.append(Token(tt_div, ps=self.pos))
                self.advance()
            elif self.current_char == '^':
                tokens.append(Token(tt_pow, ps=self.pos))
                self.advance()
            elif self.current_char == '(':
                tokens.append(Token(tt_lparen, ps=self.pos))
                self.advance()
            elif self.current_char == ')':
                tokens.append(Token(tt_rparen, ps=self.pos))
                self.advance()
            elif self.current_char == '!':
                tok, error = self.make_ne()
                if error: return [], error
                tokens.append(tok)
            elif self.current_char == '=':
                tokens.append(self.make_eq())
            elif self.current_char == '>':
                tokens.append(self.make_gt())
            elif self.current_char == '<':
                tokens.append(self.make_lt())
            elif self.current_char == ',':
                tokens.append(Token(tt_comma, ps=self.pos))
                self.advance()
            # elif self.current_char == '#':
            #     tokens.append(Token(tt_comment))
            else:
                ps = self.pos.copy()
                char = self.current_char
                self.advance()
                return [], IllegalCharError(ps, self.pos, "'" + char + "'")
        tokens.append(Token(tt_eof, ps=self.pos))
        return tokens, None

    def make_number(self):
        num_str = ''
        dot_count = 0
        pst = self.pos.copy()

        while self.current_char != None and self.current_char in digits + '.':
            if self.current_char == '.':
                if dot_count == 1: break
                dot_count += 1
                num_str += '.'
            else:
                num_str += self.current_char
            self.advance()
        if dot_count == 0:
            return Token(tt_int, int(num_str), ps=pst, pe=self.pos)
        else:
            return Token (tt_float, float(num_str), pst, self.pos)
    
    def make_ident(self):
        id_str = ''
        ps = self.pos.copy()
        
        while self.current_char != None and self.current_char in ld + '_':
            id_str += self.current_char
            self.advance()
        
        tok_type = tt_keyword if id_str in keywords else tt_ident
        return Token(tok_type, id_str, ps, self.pos)

    def make_min_arrow(self):
        tok_type = tt_minus
        ps = self.pos.copy()
        self.advance()

        if self.current_char == '>':
            self.advance()
            tok_type = tt_arrow
        
        return Token(tok_type, ps=ps, pe=self.pos)

    def make_ne(self):
        ps = self.pos.copy()
        self.advance()

        if self.current_char == '=':
            self.advance()
            return Token(tt_ne, ps=ps, pe=self.pos), None
        self.advance()
        return None, ExpectedCharError(ps, self.pos, "Expected '=' after '!'")
    
    def make_eq(self):
        tok_type = tt_eq
        ps = self.pos.copy()
        self.advance()

        if self.current_char == '=':
            self.advance()
            tok_type = tt_ee
        
        return Token(tok_type, ps=ps, pe=self.pos)
    
    def make_gt(self):
        tok_type = tt_gt
        ps = self.pos.copy()
        self.advance()

        if self.current_char == '=':
            self.advance()
            tok_type = tt_gte
        
        return Token(tok_type, ps=ps, pe=self.pos)
    
    def make_lt(self):
        tok_type = tt_lt
        ps = self.pos.copy()
        self.advance()

        if self.current_char == '=':
            self.advance()
            tok_type = tt_lte
        
        return Token(tok_type, ps=ps, pe=self.pos)
        
########################
# Nodes                #
########################

class NumberNode:
    def __init__(self, tok):
        self.tok = tok
        self.ps = self.tok.ps
        self.pe = self.tok.pe
    
    def __repr__(self):
        return f'{self.tok}'

class VarAccessNode:
    def __init__(self, vntok):
        self.vntok = vntok
        self.ps = self.vntok.ps
        self.pe = self.vntok.pe

class VarAssignNode:
    def __init__(self, vntok, valuenode):
        self.vntok = vntok
        self.value = valuenode
        self.ps = self.vntok.ps
        self.pe = self.value.pe

class BinOpNode:
    def __init__(self, lnode, optok, rnode):
        self.lnode = lnode
        self.optok = optok
        self.rnode = rnode

        self.ps = self.lnode.ps
        self.pe = self.rnode.pe

    def __repr__(self):
        return f'({self.lnode}, {self.optok}, {self.rnode})'

class UnaryOpNode:
    def __init__(self, optok, node):
        self.optok = optok
        self.node = node

        self.ps = self.optok.ps
        self.pe = self.node.pe
    
    def __repr__(self):
        return f'({self.optok}, {self.node})'

class IfNode:
    def __init__(self, cases, else_case):
        self.cases = cases
        self.else_case = else_case

        self.ps = self.cases[0][0].ps
        self.pe = (self.else_case or self.cases[len(self.cases)-1][0]).pe

class ForNode:
    def __init__(self, varnametok, startvalnode, endvalnode, stepvalnode, bodynode):
        self.varnametok = varnametok
        self.startvalnode = startvalnode
        self.endvalnode = endvalnode
        self.stepvalnode = stepvalnode
        self.bodynode = bodynode

        self.ps = self.varnametok.ps
        self.pe = self.bodynode.pe

class WhileNode:
    def __init__(self, conditionnode, bodynode):
        self.conditionnode = conditionnode
        self.bodynode = bodynode

        self.ps = self.conditionnode.ps
        self.pe = self.bodynode.pe

class FuncDefNode:
    def __init__(self, varnametok, argnametoks, bodynode):
        self.varnametok = varnametok
        self.argnametoks = argnametoks
        self.bodynode = bodynode

        if self.varnametok:
            self.ps = self.varnametok.ps
        elif len(self.argnametoks) > 0:
            self.ps = self.argnametoks[0].ps
        else:
            self.ps = self.bodynode.ps
        
        self.pe = self.bodynode.pe

class CallNode:
    def __init__(self, nodetocall, argnodes):
        self.nodetocall = nodetocall
        self.argnodes = argnodes

        self.ps = self.nodetocall.ps

        if len(self.argnodes) > 0:
            self.pe = self.argnodes[len(self.argnodes) - 1].pe
        else:
            self.pe = self.nodetocall.pe

########################
# Parse Result         #
########################

class ParseResult:
    def __init__(self):
        self.error = None
        self.node = None
        self.advance_cnt = 0
    
    def register_advance(self):
        self.advance_cnt += 1

    def register(self, res):
        self.advance_cnt += res.advance_cnt
        if res.error: self.error = res.error
        return res.node


    def success(self, node):
        self.node = node
        return self
    
    def failure(self, error):
        if not self.error or self.advance_cnt == 0:
            self.error = error
        return self

########################
# Parser               #
########################

class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.tokidx = -1
        self.current_tok = None
        self.advance()
    
    def advance(self):
        self.tokidx += 1
        if self.tokidx < len(self.tokens):
            self.current_tok = self.tokens[self.tokidx]

    #############################

    def binop(self, funca, ops, funcb=None):
        if funcb == None: funcb = funca
        res = ParseResult()
        left = res.register(funca())
        if res.error: return res

        while self.current_tok.type in ops or (self.current_tok.type, self.current_tok.value) in ops:
            optok = self.current_tok
            res.register_advance()
            self.advance()
            right = res.register(funcb())
            if res.error: return res
            left = BinOpNode(left, optok, right)
        
        return res.success(left)
    
    #############################

    def if_expr(self):
        res = ParseResult()
        cases = []
        else_case = None

        if not self.current_tok.matches(tt_keyword, 'if'):
            return res.failure(InvalidSyntaxError(self.current_tok.ps, self.current_tok.pe, "Expected 'if'"))
        
        res.register_advance()
        self.advance()

        condition = res.register(self.expr())
        if res.error: return res 

        if not self.current_tok.matches(tt_keyword, 'then'):
            return res.failure(InvalidSyntaxError(self.current_tok.ps, self.current_tok.pe, "Expected 'then'"))
        
        res.register_advance()
        self.advance()

        expr = res.register(self.expr())
        if res.error: return res 
        cases.append((condition, expr))

        while self.current_tok.matches(tt_keyword, 'elif'):
            res.register_advance()
            self.advance()

            condition = res.register(self.expr())
            if res.error: return res

            if not self.current_tok.matches(tt_keyword, 'then'):
                return res.failure(InvalidSyntaxError(self.current_tok.ps, self.current_tok.pe, "Expected 'then'"))
            
            res.register_advance()
            self.advance()

            expr = res.register(self.expr())
            if res.error: return res
            cases.append((condition, expr))
        
        if self.current_tok.matches(tt_keyword, 'else'):
            res.register_advance()
            self.advance()

            else_case = res.register(self.expr())
            if res.error: return res
            
        
        return res.success(IfNode(cases, else_case))
    
    def for_expr(self):
        res = ParseResult()
        if not self.current_tok.matches(tt_keyword, 'for'):
            return res.failure(InvalidSyntaxError(self.current_tok.ps, self.current_tok.pe, "Expected 'for'"))
        
        res.register_advance()
        self.advance()

        if self.current_tok.type != tt_ident:
            return res.failure(InvalidSyntaxError(self.current_tok.ps, self.current_tok.pe, "Expected variable name"))
        
        var_name = self.current_tok
        res.register_advance()
        self.advance()

        if self.current_tok.type != tt_eq:
            return res.failure(InvalidSyntaxError(self.current_tok.ps, self.current_tok.pe, "Expected '='"))
        
        res.register_advance()
        self.advance()

        start_val = res.register(self.expr())
        if res.error: return res 

        if not self.current_tok.matches(tt_keyword, 'to'):
            return res.failure(InvalidSyntaxError(self.current_tok.ps, self.current_tok.pe, "Expected 'to'"))
        
        res.register_advance()
        self.advance()

        end_val = res.register(self.expr())
        if res.error: return res

        if self.current_tok.matches(tt_keyword, 'step'):
            res.register_advance()
            self.advance()

            step_val = res.register(self.expr())
            if res.error: return res

        else:
            step_val = None 
        
        if not self.current_tok.matches(tt_keyword, 'then'):
            return res.failure(InvalidSyntaxError(self.current_tok.ps, self.current_tok.pe, "Expected 'then'"))
        
        res.register_advance()
        self.advance()

        body = res.register(self.expr())
        if res.error: return res

        return res.success(ForNode(var_name, start_val, end_val, step_val, body))
    
    def while_expr(self):
        res = ParseResult()

        if not self.current_tok.matches(tt_keyword, 'while'):
            return res.failure(InvalidSyntaxError(self.current_tok.ps, self.current_tok.pe, "Expected 'while'"))
        
        res.register_advance()
        self.advance()

        condition = res.register(self.expr())
        if res.error: return res

        if not self.current_tok.matches(tt_keyword, 'then'):
            return res.failure(InvalidSyntaxError(self.current_tok.ps, self.current_tok.pe, "Expected 'then'"))

        res.register_advance()
        self.advance()

        body = res.register(self.expr())
        if res.error: return res

        return res.success(WhileNode(condition, body))

    def call(self):
        res = ParseResult()
        atom = res.register(self.atom())
        if res.error: return res

        if self.current_tok.type == tt_lparen:
            res.register_advance()
            self.advance()
            argnodes = []

            if self.current_tok.type == tt_rparen:
                res.register_advance()
                self.advance()
            else:
                argnodes.append(res.register(self.expr()))
                if res.error: return res.failure(InvalidSyntaxError(self.current_tok.ps, self.current_tok.pe, "Expected ')', 'variable', integer, decimal, variable name, '+', '-', or '('"))

                while self.current_tok.type == tt_comma:
                    res.register_advance()
                    self.advance()

                    argnodes.append(res.register(self.expr()))
                    if res.error: return res

                if self.current_tok.type != tt_rparen:
                    return res.failure(InvalidSyntaxError(self.current_tok.ps, self.current_tok.pe, "Expected ',' or ')'"))
                
                res.register_advance()
                self.advance()
            return res.success(CallNode(atom, argnodes))
        return res.success(atom)

    def atom(self):
        res = ParseResult()
        tok = self.current_tok

        if tok.type in (tt_int, tt_float):
            res.register_advance()
            self.advance()
            return res.success(NumberNode(tok))
        
        elif tok.type == tt_ident:
            res.register_advance()
            self.advance()
            return res.success(VarAccessNode(tok))
        
        elif tok.type == tt_lparen:
            res.register_advance()
            self.advance()
            expr = res.register(self.expr())
            if res.error: return res.error
            if self.current_tok.type == tt_rparen:
                res.register(self.advance)
                return res.success(expr)
            else:
                return res.failure(InvalidSyntaxError(self.current_tok.ps, self.current_tok.pe, "Expected ')'"))
        
        elif tok.matches(tt_keyword, 'if'):
            if_expr = res.register(self.if_expr())
            if res.error: return res
            return res.success(if_expr)
        
        elif tok.matches(tt_keyword, 'for'):
            for_expr = res.register(self.for_expr())
            if res.error: return res
            return res.success(for_expr)
        
        elif tok.matches(tt_keyword, 'while'):
            while_expr = res.register(self.while_expr())
            if res.error: return res
            return res.success(while_expr)
        
        elif tok.matches(tt_keyword, 'func'):
            funcdef = res.register(self.funcdef())
            if res.error: return res
            return res.success(funcdef)
        
        return res.failure(InvalidSyntaxError(tok.ps, tok.pe, "Expected integer, decimal, variable name, '+', '-', '(', 'if', 'for', 'while', or 'func'"))

    def power(self):
        return self.binop(self.call, (tt_pow, ), self.factor)

    def factor(self):
        res = ParseResult()
        tok = self.current_tok

        if tok.type in (tt_plus, tt_minus):
            res.register_advance()
            factor = res.register(self.factor())
            if res.error: return res
            return UnaryOpNode(tok, factor)

        return self.power()

    def term(self):
        return self.binop(self.factor, (tt_mult, tt_div))

    def arith_expr(self):
        return self.binop(self.term, (tt_plus, tt_minus))

    def comp_expr(self):
        res = ParseResult()
        
        if self.current_tok.matches(tt_keyword, 'not'):
            optok = self.current_tok
            res.register_advance()
            self.advance()

            node = res.register(self.com_expr())
            if res.error: return res
            return res.success(UnaryOpNode(optok, node))
        
        node = res.register(self.binop(self.arith_expr, (tt_ee, tt_ne, tt_gt, tt_lt, tt_gte, tt_lte)))

        if res.error: return res.failure(InvalidSyntaxError(self.current_tok.ps, self.current_tok.pe, "Expected integer, decimal, variable name, '+', '-', '(', or 'not'"))

        return res.success(node)

    def expr(self):
        res = ParseResult()

        if self.current_tok.matches(tt_keyword, 'variable'):
            res.register_advance()

            if self.current_tok.type != tt_ident:
                return res.failure(InvalidSyntaxError(self.current_tok.ps, self.current_tok.pe, 'Expected identifier'))
            
            var_name = self.current_tok
            res.register_advance()
            if self.current_tok.type != tt_eq:
                return res.failure(InvalidSyntaxError(self.current_tok.ps, self.current_tok.pe, "Expected '='"))
            res.register_advance()
            expr = res.register(self.expr())
            if res.error: return res
            return res.success(VarAssignNode(var_name, expr))

        node = res.register(self.binop(self.comp_expr, ((tt_keyword,'and'), (tt_keyword, 'or'))))
        if res.error: 
            return res.failure(InvalidSyntaxError(self.current_tok.ps, self.current_tok.pe, "Expected 'variable', 'if', 'for', 'while', 'func', integer, decimal, variable name, '+', '-', '(', or 'not'"))

        return res.success(node)

    def funcdef(self):
        res = ParseResult()

        if not self.current_tok.matches(tt_keyword, "func"):
            return res.failure(InvalidSyntaxError(self.current_tok.ps, self.current_tok.pe, "Expected 'func'"))

        res.register_advance()

        if self.current_tok.type == tt_ident:
            varnametok = self.current_tok
            res.register_advance()
            self.advance()
            if self.current_tok.type != tt_lparen:
                return res.failure(InvalidSyntaxError(self.current_tok.ps, self.current_tok.pe, "Expected '('"))
        else:
            varnametok = None
            if self.current_tok.type != tt_lparen:
                return res.failure(InvalidSyntaxError(self.current_tok.ps, self.current_tok.pe, "Expected identifier or '('"))
        
        res.register_advance()
        self.advance()
        argnametoks = []
        
        if self.current_tok.type == tt_ident:
            argnametoks.append(self.current_tok)
            res.register_advance()
            self.advance()
        
            while self.current_tok.type == tt_comma:
                res.register_advance()
                self.advance()

                if self.current_tok.type != tt_ident:
                    return res.failure(InvalidSyntaxError(self.current_tok.ps, self.current_tok.pe, "Expected identifier"))
                
                argnametoks.append(self.current_tok)
                res.register_advance()
                self.advance()

            if self.current_tok.type != tt_rparen:
                return res.failure(InvalidSyntaxError(self.current_tok.ps, self.current_tok.pe, "Expected ',' or ')'"))
        else:
            if self.current_tok.type != tt_rparen:
                return res.failure(InvalidSyntaxError(self.current_tok.ps, self.current_tok.pe, "Expected identifier or ')'"))
        
        res.register_advance()
        self.advance()

        if self.current_tok.type == tt_arrow:
            return res.failure(InvalidSyntaxError(self.current_tok.ps, self.current_tok.pe, "Expected '->'"))
        
        res.register_advance()
        self.advance()
        nodetoreturn = res.register(self.expr())
        if res.error: return res 

        return res.success(FuncDefNode(varnametok, argnametoks, nodetoreturn))
             
            

    
    #############################

    def parse(self):
        res = self.expr()
        if not res.error and self.current_tok.type != tt_eof:
            return res.failure(InvalidSyntaxError(self.current_tok.ps, self.current_tok.pe, "Expected '+', '-', '*', or '/'"))
        return res

########################
# Runtime Result       #
########################

class RTResult:
    def __init__(self):
        self.value = None 
        self.error = None
    
    def register(self, res):
        if res.error: self.error = res.error
        return res.value
    
    def success(self, value):
        self.value = value
        return self
    
    def failure(self, error):
        self.error = error
        return self

########################
# Values               #
########################
class Value:
	def __init__(self):
		self.set_pos()
		self.set_context()

	def set_pos(self, pos_start=None, pos_end=None):
		self.pos_start = pos_start
		self.pos_end = pos_end
		return self

	def set_context(self, context=None):
		self.context = context
		return self

	def added_to(self, other):
		return None, self.illegal_operation(other)

	def subbed_by(self, other):
		return None, self.illegal_operation(other)

	def multed_by(self, other):
		return None, self.illegal_operation(other)

	def dived_by(self, other):
		return None, self.illegal_operation(other)

	def powed_by(self, other):
		return None, self.illegal_operation(other)

	def get_comparison_eq(self, other):
		return None, self.illegal_operation(other)

	def get_comparison_ne(self, other):
		return None, self.illegal_operation(other)

	def get_comparison_lt(self, other):
		return None, self.illegal_operation(other)

	def get_comparison_gt(self, other):
		return None, self.illegal_operation(other)

	def get_comparison_lte(self, other):
		return None, self.illegal_operation(other)

	def get_comparison_gte(self, other):
		return None, self.illegal_operation(other)

	def anded_by(self, other):
		return None, self.illegal_operation(other)

	def ored_by(self, other):
		return None, self.illegal_operation(other)

	def notted(self, other):
		return None, self.illegal_operation(other)

	def execute(self, args):
		return RTResult().failure(self.illegal_operation())

	def copy(self):
		raise Exception('No copy method defined')

	def is_true(self):
		return False

	def illegal_operation(self, other=None):
		if not other: other = self
		return RTError(
			self.pos_start, other.pos_end,
			'Illegal operation',
			self.context
		)

class Number(Value):
    def __init__(self, value):
        super().__init__()
        self.value = value
    
    def added_to(self, other):
        if isinstance(other, Number):
            return Number(self.value + other.value).set_context(self.context), None
    
    def subbed_by(self, other):
        if isinstance(other, Number):
            return Number(self.value - other.value).set_context(self.context), None

    def multed_by(self, other):
        if isinstance(other, Number):
            return Number(self.value * other.value).set_context(self.context), None
    
    def dived_by(self, other):
        if isinstance(other, Number):
            if other.value == 0:
                return None, RTError(other.ps, other.pe, "Division by 0", self.context)
            return Number(self.value / other.value).set_context(self.context), None
    
    def powed_by(self, other):
        if isinstance(other, Number):
            return Number(self.value ** other.value).set_context(self.context), None
    
    def cmpe(self, other):
        if isinstance(other, Number):
            return Number(int(self.value == other.value)).set_context(self.context), None
    
    def cmpne(self, other):
        if isinstance(other, Number):
            return Number(int(self.value != other.value)).set_context(self.context), None
    
    def cmpgt(self, other):
        if isinstance(other, Number):
            return Number(int(self.value > other.value)).set_context(self.context), None
    
    def cmplt(self, other):
        if isinstance(other, Number):
            return Number(int(self.value < other.value)).set_context(self.context), None
    
    def cmpgte(self, other):
        if isinstance(other, Number):
            return Number(int(self.value >= other.value)).set_context(self.context), None
    
    def cmplte(self, other):
        if isinstance(other, Number):
            return Number(int(self.value <= other.value)).set_context(self.context), None
    
    def anded_by(self, other):
        if isinstance(other, Number):
            return Number(int(self.value and other.value)).set_context(self.context), None
    
    def ored_by(self, other):
        if isinstance(other, Number):
            return Number(int(self.value or other.value)).set_context(self.context), None
    
    def notted(self):
        return Number(1 if self.value == 0 else 0).set_context(self.context), None
    
    def is_true(self):
        return self.value != 0
    
    def copy(self):
        copy = Number(self.value)
        copy.set_pos(self.ps, self.pe)
        copy.set_context(self.context)
        return copy

    def __repr__(self):
        return str(self.value)

class Function(Value):
    def __init__(self, name, body_node, arg_names):
        super().__init__()
        self.name = name or "<anonymous>"
        self.bodynode = body_node
        self.argnames = arg_names
    
    def execute(self, args):
        res = RTResult()
        interpreter = Interpreter()
        new_ctx = Context(self.name, self.context, self.pos_start)
        new_ctx.symboltable = SymbolTable(new_ctx.parent.symboltable)

        if len(args) > len(self.argnames):
            return res.failure(RTError(self.pos_start, self.pos_end, f"{len(args) - len(self.argnames)} too many arguments passed into '{self.name}'", self.context))
        
        if len(args) < len(self.argnames):
            return res.failure(RTError(self.pos_start, self.pos_end, f"{len(self.argnames) - len(args)} too few arguments passed into '{self.name}'", self.context))
        
        for i in range(len(args)):
            argname = self.argnames[i]
            argval = args[i]
            argval.set_context(new_ctx)
            new_ctx.symboltable.set(argname, argval)
        
        value = res.register(interpreter.visit(self.bodynode, new_ctx))
        if res.error: return res
        return res.success(value)

    def copy(self):
        copy = Function(self.name, self.bodynode, self.argnames)
        copy.set_context(self.context)
        copy.set_pos(self.pos_start, self.pos_end)
        return copy
    
    def __repr__(self):
        return f"<function {self.name}>" if self.name != "<anonymous>" else "<function 'anonymous'>"


########################
# Context              #
########################

class Context:
    def __init__(self, dn, parent=None, pep=None):
        self.dn = dn
        self.parent = parent
        self.pep = pep
        self.symboltable = None

########################
# Symbol Table         #
########################

class SymbolTable:
    def __init__(self, parent=None):
        self.symbols = {}
        self.parent = parent
    
    def get(self, name):
        value = self.symbols.get(name, None)
        if value == None and self.parent:
            return self.parent.get(name)
        return value
    
    def set(self, name, value):
        self.symbols[name] = value
    
    def remove(self, name):
        del self.symbols[name]

########################
# Interpreter          #
########################

class Interpreter:
    def visit(self, node, context):
        methodname = f'visit_{type(node).__name__}'
        method = getattr(self, methodname, self.no_visit_method)
        return method(node, context)

    def no_visit_method(self, node, context):
        raise Exception(f"No visit_{type(node).__name__} method defined")
    
    #############################

    def visit_NumberNode(self, node, context):
        return RTResult().success(Number(node.tok.value).set_context(context).set_pos(node.ps, node.pe))
    
    def visit_VarAccessNode(self, node, context):
        res = RTResult()
        variable = node.vntok.value
        value = context.symboltable.get(variable)

        if not value:
            return res.failure(RTError(node.ps, node.pe,f"'{variable}' is not defined"), context)
        value = value.copy().set_pos(node.ps, node.pe)
        return res.success(value)
    
    def visit_VarAssignNode(self, node, context):
        res = RTResult()
        variable = node.vntok.value
        value = res.register(self.visit(node.value, context))
        if res.error: return res

        context.symboltable.set(variable, value)
        return res.success(value)

    def visit_BinOpNode(self, node, context):
        res = RTResult()
        left = res.register(self.visit(node.lnode, context))
        if res.error: return res
        right = res.register(self.visit(node.rnode, context))
        if res.error: return res
        
        error = None

        if node.optok.type == tt_plus:
            result, error = left.added_to(right)
        elif node.optok.type == tt_minus:
            result, error = left.subbed_by(right)
        elif node.optok.type == tt_mult:
            result, error = left.multed_by(right)
        elif node.optok.type == tt_div:
            result, error = left.dived_by(right)
        elif node.optok.type == tt_pow:
            result, error = left.powed_by(right)
        elif node.optok.type == tt_ee:
            result, error = left.cmpe(right)
        elif node.optok.type == tt_ne:
            result, error = left.cmpne(right)
        elif node.optok.type == tt_gt:
            result, error = left.cmpgt(right)
        elif node.optok.type == tt_lt:
            result, error = left.cmplt(right)
        elif node.optok.type == tt_gte:
            result, error = left.cmpgte(right)
        elif node.optok.type == tt_lte:
            result, error = left.cmplte(right)
        elif node.optok.matches(tt_keyword, 'and'):
            result, error = left.anded_by(right)
        elif node.optok.matches(tt_keyword, 'or'):
            result, error = left.ored_by(right)
        
        
        if error:
            return res.failure(error)

        return res.success(result.set_pos(node.ps, node.pe))
    
    def visit_UnaryOpNode(self, node, context):
        res = RTResult()
        number = res.register(self.visit(node.node, context))
        if res.error: return res
        error = None
        if node.optok.type == tt_minus:
            number, error = number.multed_by(Number(-1))
        if node.optok.matches(tt_keyword, 'not'):
            number, error = number.notted()
        
        if error: res.failure(error)
        return res.success(number.set_pos(node.ps, node.pe))
    
    def visit_IfNode(self, node, context):
        res = RTResult()

        for condition, expr in node.cases:
            condition_value = res.register(self.visit(condition, context))
            if res.error: return res 

            if condition_value.is_true():
                expr_value = res.register(self.visit(expr, context))
                if res.error: return res
                return res.success(expr_value)
        
        if node.else_case:
            else_value = res.register(self.visit(node.else_case, context))
            if res.error: return res
            return res.success(else_value)
        
        return res.success(None)
    
    def visit_ForNode(self, node, context):
        res = RTResult()

        start_val = res.register(self.visit(node.startvalnode, context))
        if res.error: return res

        end_val = res.register(self.visit(node.endvalnode, context))
        if res.error: return res

        if node.stepvalnode:
            step_val = res.register(self.visit(node.stepvalnode, context))
            if res.error: return res 
        else:
            step_val = Number(1)

        i = start_val.value

        if step_val >= 0:
            condition = lambda: i < end_val.value
        else:
            condition = lambda: i > end_val.value
        
        while condition():
            context.symboltable.set(node.varnametok.value, Number(i))
            i += 1

            res.register(self.visit(node.bodynode, context))
            if res.error: return res
        
        return res.success(None)

    def visit_WhileNode(self, node, context):
        res = RTResult()

        while True:
            condition = res.register(self.visit(node.conditionnode, context))
            if res.error: return res

            if not condition.is_true: break

            res.register(self.visit(node.bodynode, context))
            if res.error:return res
        return res.success(None)
    
    def visit_FuncDefNode(self, node, context):
        res = RTResult()

        funcname = node.varnametok.value if node.varnametok else None
        bodynode = node.bodynode
        argnames = [argname.value for argname in node.argnametoks]
        funcval = Function(funcname, bodynode, argnames).set_context(context).set_pos(node.ps, node.pe)
        if node.varnametok:
            context.symboltable.set(funcname, funcval)
        
        return funcval

    def visit_CallNode(self, node, context):
        res = RTResult()
        args = []

        valuetocall = res.register(self.visit(node.nodetocall, context))
        if res.error: return res
        valuetocall = valuetocall.copy().set_pos(node.ps, node.pe)

        for argnode in node.argnodes:
            args.append(res.register(self.visit(argnode, context)))
            if res.error: return res
        returnval = res.register(valuetocall.execute(args))
        if res.error: return res
        return returnval


########################
# Run                  #
########################

##########
# Global Symbol Table
##########
gst = SymbolTable()
gst.set("null", Number(0))
gst.set("true", Number(1))
gst.set("false", Number(0))

def run(fn, text):
    # generate tokens
    lexer = Lexer(fn, text)
    tokens, error = lexer.make_tokens()

    if error: return None, error

    # generate abstract syntax tree (ast)
    parser = Parser(tokens)
    ast = parser.parse()

    if ast.error: return None, ast.error

    # run program
    interpreter = Interpreter()
    context = Context('<program>')
    context.symboltable = gst
    result = interpreter.visit(ast.node, context)


    return result.value, result.error